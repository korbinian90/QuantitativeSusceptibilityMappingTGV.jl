function rotate3D(image::AbstractArray{T,N}, angle_x, mode=Linear()) where {T,N}
    centered = OffsetArrays.centered(image)
    R = RotX(deg2rad(angle_x))
    rotated_points = [R * p for p in CartesianIndices(centered)]
    return resample_new_points(centered, rotated_points, mode)
end

Base.:*(R::Rotation, i::CartesianIndex) = R * SVector(Tuple(i))

function resample_new_points(array::AbstractArray{T,N}, new_points, mode) where {T,N}
    itp = interpolate(array, BSpline(mode))
    safe_apply(r) =
    if checkbounds(Bool, array, r...)
        itp(r...)
    else
        zero(T)
    end
    return T.(safe_apply.(new_points))
end
