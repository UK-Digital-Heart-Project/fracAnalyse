import numpy as np

def xform_nii(nii, tolerance=0.1, preferredForm='s'):
    # Save a copy of the header as it was loaded
    nii['original']['hdr'] = nii['hdr']

    if tolerance <= 0:
        tolerance = np.finfo(float).eps

    if preferredForm not in ['s', 'q', 'S', 'Q']:
        preferredForm = 's'

    # Apply scl_slope and scl_inter if needed
    if nii['hdr']['dime']['scl_slope'] != 0 and \
            nii['hdr']['dime']['datatype'] in [2, 4, 8, 16, 64, 256, 512, 768] and \
            (nii['hdr']['dime']['scl_slope'] != 1 or nii['hdr']['dime']['scl_inter'] != 0):

        nii['img'] = nii['hdr']['dime']['scl_slope'] * nii['img'].astype(float) + nii['hdr']['dime']['scl_inter']

        if nii['hdr']['dime']['datatype'] == 64:
            nii['hdr']['dime']['datatype'] = 64
            nii['hdr']['dime']['bitpix'] = 64
        else:
            nii['img'] = nii['img'].astype(np.float32)
            nii['hdr']['dime']['datatype'] = 16
            nii['hdr']['dime']['bitpix'] = 32

        nii['hdr']['dime']['glmax'] = np.max(nii['img'])
        nii['hdr']['dime']['glmin'] = np.min(nii['img'])
        nii['hdr']['dime']['scl_slope'] = 0

    if nii['hdr']['dime']['scl_slope'] != 0 and \
            nii['hdr']['dime']['datatype'] in [32, 1792]:

        nii['img'] = nii['hdr']['dime']['scl_slope'] * nii['img'].astype(float) + nii['hdr']['dime']['scl_inter']

        if nii['hdr']['dime']['datatype'] == 32:
            nii['img'] = nii['img'].astype(np.float32)

        nii['hdr']['dime']['glmax'] = np.max(nii['img'])
        nii['hdr']['dime']['glmin'] = np.min(nii['img'])
        nii['hdr']['dime']['scl_slope'] = 0

    if nii['filetype'] == 0:
        if os.path.exists(f"{nii['fileprefix']}.mat"):
            M = load_mat(f"{nii['fileprefix']}.mat")
            R = M[:3, :3]
            T = M[:3, 3]
            T = R @ np.ones(3) + T
            M[:3, 3] = T
            nii['hdr']['hist']['qform_code'] = 0
            nii['hdr']['hist']['sform_code'] = 1
            nii['hdr']['hist']['srow_x'] = M[0, :]
            nii['hdr']['hist']['srow_y'] = M[1, :]
            nii['hdr']['hist']['srow_z'] = M[2, :]
        else:
            nii['hdr']['hist']['rot_orient'] = []
            nii['hdr']['hist']['flip_orient'] = []
            return nii

    hdr = nii['hdr']
    hdr, orient = change_hdr(hdr, tolerance, preferredForm)

    if not np.array_equal(orient, [1, 2, 3]):
        old_dim = hdr['dime']['dim'][1:4]

        if len(nii['img'].shape) > 3:
            pattern = np.arange(np.prod(old_dim)).reshape(old_dim)
        else:
            pattern = []

        rot_orient = (orient + 1) % 3
        flip_orient = orient - rot_orient

        for i in range(3):
            if flip_orient[i]:
                if pattern:
                    pattern = np.flip(pattern, axis=i)
                else:
                    nii['img'] = np.flip(nii['img'], axis=i)

        tmp, rot_orient = np.sort(rot_orient), np.argsort(rot_orient)

        new_dim = old_dim[rot_orient]
        hdr['dime']['dim'][1:4] = new_dim

        new_pixdim = hdr['dime']['pixdim'][1:4]
        new_pixdim = new_pixdim[rot_orient]
        hdr['dime']['pixdim'][1:4] = new_pixdim

        tmp = hdr['hist']['originator'][0:3]
        tmp = tmp[rot_orient]
        flip_orient = flip_orient[rot_orient]

        for i in range(3):
            if flip_orient[i] and tmp[i] != 0:
                tmp[i] = new_dim[i] - tmp[i] + 1

        hdr['hist']['originator'][0:3] = tmp
        hdr['hist']['rot_orient'] = rot_orient
        hdr['hist']['flip_orient'] = flip_orient

        if pattern:
            pattern = np.transpose(pattern, rot_orient)
            pattern = pattern.flatten()

            if hdr['dime']['datatype'] in [32, 1792, 128, 511]:
                tmp = nii['img'][..., 0].reshape((np.prod(new_dim), *hdr['dime']['dim'][4:]))
                tmp = tmp[pattern, :]
                nii['img'][..., 0] = tmp.reshape((*new_dim, *hdr['dime']['dim'][4:]))

                tmp = nii['img'][..., 1].reshape((np.prod(new_dim), *hdr['dime']['dim'][4:]))
                tmp = tmp[pattern, :]
                nii['img'][..., 1] = tmp.reshape((*new_dim, *hdr['dime']['dim'][4:]))

                if hdr['dime']['datatype'] in [128, 511]:
                    tmp = nii['img'][..., 2].reshape((np.prod(new_dim), *hdr['dime']['dim'][4:]))
                    tmp = tmp[pattern, :]
                    nii['img'][..., 2] = tmp.reshape((*new_dim, *hdr['dime']['dim'][4:]))
            else:
                nii['img'] = nii['img'].reshape((np.prod(new_dim), *hdr['dime']['dim'][4:]))
                nii['img'] = nii['img'][pattern, :]
                nii['img'] = nii['img'].reshape((*new_dim, *hdr['dime']['dim'][4:]))
        else:
            if hdr['dime']['datatype'] in [32, 1792, 128, 511]:
                nii['img'][..., 0] = np.transpose(nii['img'][..., 0], rot_orient)
                nii['img'][..., 1] = np.transpose(nii['img'][..., 1], rot_orient)

                if hdr['dime']['datatype'] in [128, 511]:
                    nii['img'][..., 2] = np.transpose(nii['img'][..., 2], rot_orient)
            else:
                nii['img'] = np.transpose(nii['img'], rot_orient)

    else:
        hdr['hist']['rot_orient'] = []
        hdr['hist']['flip_orient'] = []

    nii['hdr'] = hdr
    return nii

def change_hdr(hdr, tolerance, preferredForm):
    orient = np.array([1, 2, 3])
    affine_transform = True

    useForm = None

    if preferredForm == 'S':
        if hdr['hist']['sform_code'] == 0:
            raise ValueError('User requires sform, sform not set in header')
        else:
            useForm = 's'

    if preferredForm == 'Q':
        if hdr['hist']['qform_code'] == 0:
            raise ValueError('User requires qform, qform not set in header')
        else:
            useForm = 'q'

    if preferredForm == 's':
        if hdr['hist']['sform_code'] > 0:
            useForm = 's'
        elif hdr['hist']['qform_code'] > 0:
            useForm = 'q'

    if preferredForm == 'q':
        if hdr['hist']['qform_code'] > 0:
            useForm = 'q'
        elif hdr['hist']['sform_code'] > 0:
            useForm = 's'

    if useForm == 's':
        R = np.array([hdr['hist']['srow_x'][:3],
                      hdr['hist']['srow_y'][:3],
                      hdr['hist']['srow_z'][:3]])

        T = np.array([hdr['hist']['srow_x'][3],
                      hdr['hist']['srow_y'][3],
                      hdr['hist']['srow_z'][3]])

        if np.linalg.det(R) == 0 or not np.array_equal(R[R != 0], np.sum(R, axis=0)):
            hdr['hist']['old_affine'] = np.vstack([np.hstack([R, T[:, None]]), [0, 0, 0, 1]])
            R_sort = np.sort(np.abs(R.flatten()))
            R[np.abs(R) < tolerance * np.min(R_sort[-3:])] = 0
            hdr['hist']['new_affine'] = np.vstack([np.hstack([R, T[:, None]]), [0, 0, 0, 1]])

            if np.linalg.det(R) == 0 or not np.array_equal(R[R != 0], np.sum(R, axis=0)):
                raise ValueError('Non-orthogonal rotation or shearing found inside the affine matrix')

    elif useForm == 'q':
        b, c, d = hdr['hist']['quatern_b'], hdr['hist']['quatern_c'], hdr['hist']['quatern_d']
        if 1.0 - (b ** 2 + c ** 2 + d ** 2) < 0:
            if abs(1.0 - (b ** 2 + c ** 2 + d ** 2)) < 1e-5:
                a = 0
            else:
                raise ValueError('Incorrect quaternion values in this NIFTI data')
        else:
            a = np.sqrt(1.0 - (b ** 2 + c ** 2 + d ** 2))

        qfac = hdr['dime']['pixdim'][0]
        qfac = 1 if qfac == 0 else qfac
        i, j, k = hdr['dime']['pixdim'][1:4]
        k *= qfac

        R = np.array([
            [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
            [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
            [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b]
        ])

        T = np.array([hdr['hist']['qoffset_x'], hdr['hist']['qoffset_y'], hdr['hist']['qoffset_z']])

        if np.linalg.det(R) == 0 or not np.array_equal(R[R != 0], np.sum(R, axis=0)):
            hdr['hist']['old_affine'] = np.vstack([np.hstack([R * np.diag([i, j, k]), T[:, None]]), [0, 0, 0, 1]])
            R_sort = np.sort(np.abs(R.flatten()))
            R[np.abs(R) < tolerance * np.min(R_sort[-3:])] = 0
            R = R * np.diag([i, j, k])
            hdr['hist']['new_affine'] = np.vstack([np.hstack([R, T[:, None]]), [0, 0, 0, 1]])

            if np.linalg.det(R) == 0 or not np.array_equal(R[R != 0], np.sum(R, axis=0)):
                raise ValueError('Non-orthogonal rotation or shearing found inside the affine matrix')

        else:
            R = R * np.diag([i, j, k])

    else:
        affine_transform = False

    if affine_transform:
        voxel_size = np.abs(np.sum(R, axis=0))
        inv_R = np.linalg.inv(R)
        originator = inv_R @ (-T) + 1
        orient = get_orient(inv_R)

        hdr['dime']['pixdim'][1:4] = voxel_size
        hdr['hist']['originator'][0:3] = originator

        hdr['hist']['qform_code'] = 0
        hdr['hist']['sform_code'] = 0

    space_unit, time_unit = get_units(hdr)

    if space_unit != 1:
        hdr['dime']['pixdim'][1:4] *= space_unit
        hdr['dime']['xyzt_units'] = np.bitwise_or(np.bitwise_and(hdr['dime']['xyzt_units'], 248), 10)

    hdr['dime']['pixdim'] = np.abs(hdr['dime']['pixdim'])

    return hdr, orient

def get_orient(R):
    orient = []

    for i in range(3):
        index = np.argmax(np.abs(R[i, :])) + 1
        sign = np.sign(np.sum(R[i, :]))
        orient.append(index * sign)

    orient = [int(x) for x in orient]

    for i in range(3):
        if orient[i] < 0:
            orient[i] = 7 - abs(orient[i])

    return orient

def get_units(hdr):
    xyzt_units = hdr['dime']['xyzt_units']
    space_unit_code = xyzt_units & 7
    time_unit_code = xyzt_units & 56

    if space_unit_code == 1:
        space_unit = 1e3
    elif space_unit_code == 3:
        space_unit = 1e-3
    else:
        space_unit = 1

    if time_unit_code == 16:
        time_unit = 1e-3
    elif time_unit_code == 24:
        time_unit = 1e-6
    else:
        time_unit = 1

    return space_unit, time_unit
