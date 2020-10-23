module MusicTransformer

include("transformer.jl")
include("positional_encoding.jl")

export Attention, Transformer, Encoder, Decoder,
        get_positional_encoding

end # module
