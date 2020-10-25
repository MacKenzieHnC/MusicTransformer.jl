using Flux
using TensorCast

## Attention

#=
    The attention module is made of 5 dense layers.
    The first 4 linearly project the input data and position data
    so they can be processed by later parts of the module.

    The last dense layer combines the attention calculations from the different
    heads into the module's final output

    Position data is currently assumed to be relative. It accepts a tensor
    of size (word_size, 2 * num_timesteps - 1, 2 * num_instruments - 1)

    I'm struggling with a good explanation for this, but basically the
    pos_data is in the form [..., -3, -2, -1, 0, 1, 2, 3, ...] for however
    long the sequence is. Then, when we're looking at the (i,j)th query,
    we only consider the range in the position data that applies.
    e.g. if we're looking at the last note in the song, there is no
    timestep that is 1 more than that, so we only consider the positions
    from [-n, 0].

    And instead of just using the distances, we use some embedding
    with dimension (word_size) for each possible distance.

    I use the embedding from the original paper, but this file is
    indifferent to the embedding, as long as it meets the other requirements.
=#

struct Attention{
    S<:AbstractArray,
    T<:AbstractArray,
    U<:AbstractArray,
    V<:AbstractArray,
    W<:AbstractArray,
}
    W_Q::S # Dense layer for queries
    W_K::T # Dense layer for keys
    W_V::U # Dense layer for values
    W_P::V # Dense layer for positions
    W_H::W # Dense layer for heads
end

function Attention(
    word_size::Integer,
    latent_size::Integer;
    head_count = 8::Integer,
    initW = Flux.glorot_uniform,
)
    return Attention(
        initW(head_count, latent_size, word_size),
        initW(head_count, latent_size, word_size),
        initW(head_count, latent_size, word_size),
        initW(head_count, latent_size, word_size),
        initW(word_size, head_count * latent_size),
    )
end

Flux.@functor Attention

function (a::Attention)(
    query_data::AbstractArray,
    pos_data::AbstractArray;
    key_data = nothing::Union{Nothing,AbstractArray},
    mask = nothing::Union{Nothing,AbstractArray}
)
    W_Q, W_K, W_V, W_P, W_H = a.W_Q, a.W_K, a.W_V, a.W_P, a.W_H

    # For self-attention, keys and queries come from the same input
    if key_data == nothing
        key_data = query_data
    end

    # Perform the first 4 linear projections
    @reduce my_values[head,meaning,time,ins] := sum(word) W_V[head,meaning,word] * key_data[word,time,ins]
    @reduce my_keys[head,meaning,time,ins] := sum(word) W_K[head,meaning,word] * key_data[word,time,ins]
    @reduce my_pos[head,meaning,time,ins] := sum(word) W_P[head,meaning,word] * pos_data[word,time,ins]
    @reduce my_queries[head,meaning,time,ins] := sum(word) W_Q[head,meaning,word] * query_data[word,time,ins]
    
    # For every word in the query projection (i, j)
    # Add the key projection to the relevant section of the position projection
    # Then multiply by the (i, j)th query projection
    scores = []
    for i in 1:size(my_queries)[3]
        ins_scores = []
        for j in 1:size(my_queries)[4]
            Q = my_queries[:,:,i,j]
            P = my_pos[:, :, i:(i+size(my_queries)[3]-1),
                                j:(j+size(my_queries)[4]-1)]
            K = my_keys

            E = P .+ K
            @reduce C[head,q_time,q_ins,k_time,k_ins] := sum(meaning) Q[head,meaning,q_time,q_ins] * E[head,meaning,k_time,k_ins]

            # Pass the information up from the for-loop
            #   in a way that doesn't break Flux.gradient()
            # I'm sure there's a better way, but I don't know it
            if ins_scores == []
                ins_scores = C
            else
                ins_scores = cat(ins_scores, C, dims = 3)
            end
        end
        # Pass the information up again
        if scores == []
            scores = ins_scores
        else
            scores = cat(scores, ins_scores, dims=2)
        end
    end

    # Scale scores, and take the softmax over all query/key timesteps/instruments
    scaling_factor = sqrt(sum(size(my_keys, 3)))
    scores = scores ./ scaling_factor
    if mask != nothing
        scores = scores .+ mask
    end

    orig_sz = size(scores)
    scores = reshape(scores,
                        orig_sz[1],
                        orig_sz[2],
                        orig_sz[3],
                        orig_sz[4] *
                        orig_sz[5],)

    scores = softmax(scores, dims=4)
    scores = reshape(scores,
                        orig_sz[1],
                        orig_sz[2],
                        orig_sz[3],
                        orig_sz[4],
                        orig_sz[5],)

    # Get scaled values, summing along key dimensions
    @reduce scaled_values[head, meaning, q_time, q_ins] :=
        sum(k_time, k_ins) my_values[head, meaning, k_time, k_ins] *
        scores[head, q_time, q_ins, k_time, k_ins]

    # Concatenate along heads
    dims = size(scaled_values)
    scaled_values =
        reshape(scaled_values, (dims[1] * dims[2], dims[3], dims[4]))

    # Transform back to input dimensions
    @reduce z[word, time, ins] :=
        sum(long_meaning) W_H[word, long_meaning] *
        scaled_values[long_meaning, time, ins]
end

## Encoder

# An encoder is just a LayerNorm followed by an attention layer,
# With a residual connection and a relu on the end
struct Encoder
    norm::LayerNorm
    self_attn::Attention
end

function Encoder(head_count::Integer, word_size::Integer, latent_size::Integer)
    return Encoder(
        LayerNorm(word_size),
        Attention(word_size, latent_size, head_count = head_count),
    )
end

Flux.@functor Encoder

function (a::Encoder)(query_data::AbstractArray, pos_data::AbstractArray;
                        mask=nothing::Union{Nothing, AbstractArray})
    norm, self_attn = a.norm, a.self_attn
    sublayer = relu.(query_data + self_attn(norm(query_data), pos_data, mask=mask))
end

## Decoder
# A decoder is just two encoders stacked on top of each other, with the
# second one getting it's key and value data from somewhere else.

struct Decoder
    norm1::LayerNorm
    self_attn::Attention
    norm2::LayerNorm
    cross_attn::Attention
end

function Decoder(head_count::Integer, word_size::Integer, latent_size::Integer)
    return Decoder(
        LayerNorm(word_size),
        Attention(word_size, latent_size, head_count = head_count),
        LayerNorm(word_size),
        Attention(word_size, latent_size, head_count = head_count),
    )
end

Flux.@functor Decoder

function (a::Decoder)(query_data::AbstractArray,
                        key_data::AbstractArray,
                        pos_data::AbstractArray;
                        mask=nothing::Union{Nothing, AbstractArray})
    norm1, self_attn, norm2, cross_attn =
        a.norm1, a.self_attn, a.norm2, a.cross_attn

    sublayer1 = relu.(query_data + self_attn(norm1(query_data), pos_data, mask=mask))
    sublayer2 =
        relu.(sublayer1 + cross_attn(norm2(sublayer1), pos_data, key_data = norm2(key_data)))
end

## Transformer
# The transformer just brings some number of encoders and decoders together,
# with the cross-attention in the decoders getting their key and value data
# from the final output of the encoder chain.
struct Transformer
    encoder::Union{Encoder,Chain}
    decoder::Union{Decoder,Chain}
end

function Transformer(
    N::Integer,
    head_count::Integer,
    word_size::Integer,
    latent_size::Integer,
)
    return Transformer(
        Chain([Encoder(head_count, word_size, latent_size) for i = 1:N]...),
        Chain([Decoder(head_count, word_size, latent_size) for i = 1:N]...),
    )
end

Flux.@functor Transformer

function (a::Transformer)(encoder_input,
                            pos_data;
                            decoder_input = nothing,
                            mask = nothing)
    encoder, decoder = a.encoder, a.decoder

    if decoder_input == nothing
        decoder_input = encoder_input
    end

    encoded_data = encoder[1](encoder_input, pos_data, mask=mask)
    for i in 2:length(encoder)
        encoded_data = encoder[i](encoded_data, pos_data)
    end

    output = decoder[1](decoder_input, encoded_data, pos_data, mask=mask)
    for i in 2:length(decoder)
        output = decoder[i](output, encoded_data, pos_data)
    end

    return output
end
