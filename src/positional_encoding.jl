function get_positional_encoding(song::AbstractArray)

    # Positional data
    word_size = size(song)[1]
    sequence_length = size(song)[2]
    num_ins = size(song)[3]

    pos_data = [k%2==0 ? sin(1/(10000^(2*k/word_size))*t*i) : cos(1/(10000^(2*k/word_size))*t*i)
                    for k in 1:word_size,
                        t in (-sequence_length+1):(sequence_length-1),
                        i in (-num_ins+1):(num_ins-1)]

    return pos_data
end
