using Flux
using TensorCast

## Attention
struct Attention{
    S<:AbstractArray,
    T<:AbstractArray,
    U<:AbstractArray,
    V<:AbstractArray,
    W<:AbstractArray,
}
    W_Q::S
    W_K::T
    W_V::U
    W_P::V
    W_O::W
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
    key_data = nothing::Union{Nothing,AbstractArray}
)
    W_Q, W_K, W_V, W_P, W_O = a.W_Q, a.W_K, a.W_V, a.W_P, a.W_O

    # Self-attention
    if key_data == nothing
        key_data = query_data
    end

    @reduce my_values[head,meaning,time,ins] := sum(word) W_V[head,meaning,word] * key_data[word,time,ins]
    @reduce my_queries[head,meaning,time,ins] := sum(word) W_Q[head,meaning,word] * query_data[word,time,ins]
    @reduce my_keys[head,meaning,time,ins] := sum(word) W_K[head,meaning,word] * key_data[word,time,ins]
    @reduce my_pos[head,meaning,time,ins] := sum(word) W_P[head,meaning,word] * pos_data[word,time,ins]

    # Get scores
    scores = []
    for i in 1:size(my_queries)[3]
        ins_scores = []
        for j in 1:size(my_queries)[4]
            Q = my_queries[:,:,i,j]
            E = my_keys .+ my_pos[:, :, i:(i+size(my_queries)[3]-1),
                                        j:(j+size(my_queries)[4]-1)]
            @reduce C[head,q_time,q_ins,k_time,k_ins] := sum(meaning) Q[head,meaning,q_time,q_ins] * E[head,meaning,k_time,k_ins]

            # Bounce up
            if ins_scores == []
                ins_scores = C
            else
                ins_scores = cat(ins_scores, C, dims = 3)
            end
        end
        # Bounce up again
        if scores == []
            scores = ins_scores
        else
            scores = cat(scores, ins_scores, dims=2)
        end
    end

    # Scale scores, and take the softmax over all query/key timesteps/instruments
    scaling_factor = sqrt(sum(size(my_keys)))
    temp = reshape(
        scores,
        (
            size(scores)[1],
            size(scores)[2] *
            size(scores)[3] *
            size(scores)[4] *
            size(scores)[5],
        ),
    )
    scores = reshape(
        softmax(temp / scaling_factor, dims = 2),
        (
            size(scores)[1],
            size(scores)[2],
            size(scores)[3],
            size(scores)[4],
            size(scores)[5],
        ),
    )

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
        sum(long_meaning) W_O[word, long_meaning] *
        scaled_values[long_meaning, time, ins]
end

## Encoder
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

function (a::Encoder)(query_data::AbstractArray, pos_data::AbstractArray)
    norm, self_attn = a.norm, a.self_attn
    sublayer = relu.(query_data + self_attn(norm(query_data), pos_data))
end

## Decoder
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
                        pos_data::AbstractArray)
    norm1, self_attn, norm2, cross_attn =
        a.norm1, a.self_attn, a.norm2, a.cross_attn

    sublayer1 = relu.(query_data + self_attn(norm1(query_data), pos_data))
    sublayer2 =
        relu.(sublayer1 + cross_attn(norm2(sublayer1), pos_data, key_data = norm2(key_data)))
end

## Transformer

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

function (a::Transformer)(decoder_input, pos_data; encoder_input = nothing)
    encoder, decoder = a.encoder, a.decoder

    if encoder_input == nothing
        encoder_input = decoder_input
    end

    encoded_data = encoder_input
    for e in encoder
        encoded_data = e(encoded_data, pos_data)
    end

    output = decoder_input
    for d in decoder
        output = d(output, encoded_data, pos_data)
    end

    # If using onehot
    #return Flux.softmax(output, dims=1)

    return output
end
