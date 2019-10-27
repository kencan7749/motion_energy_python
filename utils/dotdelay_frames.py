import numpy as np

def dotdelay_frames(gs, scw, iin):
    # function out = dotdelay(kernel, in)
    #
    # calculate linear responses of a system kernel to an input
    #
    # INPUT: [kernel] = kernel N x D matrix where N is number of channels and
    #                   D is number of delay lines.
    # [ iin] = input N x S matrix where S is the number of samples
    #
    # OUTPUT:
    # [out] = vector of kernel responses to input
    #

    ktsize = scw.shape[1]
    ktsize2c = int(np.ceil(ktsize/2))
    ktsize2f = int(np.floor(ktsize/2))
    itsize = iin.shape[1]

    gout = np.dot(gs, iin).T
    outs = np.dot(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
    outc = - np.dot(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))

    #the first  harf
    for ii in range(ktsize2c):
        z = np.zeros([ktsize2c-ii -1 ,1])
        outs[:,ii] = np.hstack([outs[ktsize2c -ii - 1:,ii], z.flatten()])
        outc[:,ii] = np.hstack([outc[ktsize2c-ii - 1:, ii], z.flatten()])

    # the second half

    for ii in range(ktsize2f):
        ti = ii + ktsize2c
        z = np.zeros(ii + 1)
        end = len(outs)
        outs[:,ti] = np.hstack([z, outs[:end-ii - 1 ,ti]])
        outc[:,ti] = np.hstack([z, outc[:end-ii - 1, ti]])

    chouts = np.sum(outs,1)
    choutc = np.sum(outc, 1)

    return chouts, choutc