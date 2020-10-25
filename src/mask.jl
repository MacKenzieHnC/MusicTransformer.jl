using Random

function mask(region::AbstractArray, scramble_rate)
    mask = zeros(1, 1, 1, size(region)[2:end]...)
    num_masked = Integer(floor(scramble_rate * size(region)[2] * size(region)[3]))
    idx = [(i,j) for i in 1:size(region)[2], j in 1:size(region)[3]]
    idx = shuffle(idx)[1:num_masked]

    for n in idx
        mask[1,1,1,n...] = -Inf
    end

    return mask
end
