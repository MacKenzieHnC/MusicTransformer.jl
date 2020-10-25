module MusicTransformer

include("transformer.jl")
include("positional_encoding.jl")
include("mask.jl")

export Attention, Transformer, Encoder, Decoder,
        get_positional_encoding,
        mask

end # module
