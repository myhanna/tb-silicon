from __future__ import print_function

__version__='myhanna adopted with pythTB'

import numpy as np     #import some package to numerical solve
import sys             #import for text
import copy            #for deepcopying


class w90(object):
    r"""This class to make to read output file from wannier90.
    """
    def __init__(self, prefix):
        # store prefix
        self.prefix = prefix

        #read input file .win
        # read in lattice vectors
        f = open(self.prefix + ".win", "r")
        ln = f.readlines()
        f.close()
        # get lattice vector
        self.lat = np.zeros((3, 3), dtype=float)
        found = False
        for i in range(len(ln)):
            sp = ln[i].split()
            if len(sp) >= 2:
                if sp[0].lower() == "begin" and sp[1].lower() == "unit_cell_cart":
                    # get units right
                    if ln[i + 1].strip().lower() in ["ang", "angstrom"]:
                        pref = 1.0
                        skip = 1
                    else:
                        pref = 1.0
                        skip = 0
                    # now get vectors
                    for j in range(3):
                        sp = ln[i + skip + 1 + j].split()
                        for k in range(3):
                            self.lat[j, k] = float(sp[k]) * pref
                    found = True
                    break
        if not found:
            raise Exception("Unable to find unit_cell_cart block in the .win file.")

        # read in hamiltonian matrix, in eV
        f = open(self.prefix + "_hr.dat", "r")
        ln = f.readlines()
        f.close()
        #
        # get number of wannier functions
        self.num_wan = int(ln[1])
        # get number of Wigner-Seitz points
        num_ws = int(ln[2])
        # get degeneracies of Wigner-Seitz points
        deg_ws = []
        for j in range(3, len(ln)):
            sp = ln[j].split()
            for s in sp:
                deg_ws.append(int(s))
            if len(deg_ws) == num_ws:
                last_j = j
                break
            if len(deg_ws) > num_ws:
                raise Exception("Too many degeneracies for WS points!")
        deg_ws = np.array(deg_ws, dtype=int)

        # read in matrix elements
        # Convention used in w90 is to write out:
        # R1, R2, R3, i, j, ham_r(i,j,R)
        # where ham_r(i,j,R) corresponds to matrix element < i | H | j+R >

        self.ham_r = {}  # format is ham_r[(R1,R2,R3)]["h"][i,j] for < i | H | j+R >
        ind_R = 0  # which R vector in line is this?
        for j in range(last_j + 1, len(ln)):
            sp = ln[j].split()
            # get reduced lattice vector components
            ham_R1 = int(sp[0])
            ham_R2 = int(sp[1])
            ham_R3 = int(sp[2])
            # get Wannier indices
            ham_i = int(sp[3]) - 1
            ham_j = int(sp[4]) - 1
            # get matrix element
            ham_val = float(sp[5]) + 1.0j * float(sp[6])
            # store stuff, for each R store hamiltonian and degeneracy
            ham_key = (ham_R1, ham_R2, ham_R3)
            if not (ham_key in self.ham_r):
                self.ham_r[ham_key] = {
                    "h": np.zeros((self.num_wan, self.num_wan), dtype=complex),
                    "deg": deg_ws[ind_R]
                }
                ind_R += 1
            self.ham_r[ham_key]["h"][ham_i, ham_j] = ham_val

        # check if for every non-zero R there is also -R
        for R in self.ham_r:
            if not (R[0] == 0 and R[1] == 0 and R[2] == 0):
                found_pair = False
                for P in self.ham_r:
                    if not (R[0] == 0 and R[1] == 0 and R[2] == 0):
                        # check if they are opposite
                        if R[0] == -P[0] and R[1] == -P[1] and R[2] == -P[2]:
                            if found_pair:
                                raise Exception("Found duplicate negative R!")
                            found_pair = True
                if not found_pair:
                    raise Exception("Did not find negative R for R = " + R + "!")

        # read in wannier centers
        f = open(self.prefix + "_centres.xyz", "r")
        ln = f.readlines()
        f.close()
        # Wannier centers in Cartesian, Angstroms
        xyz_cen = []
        for i in range(2, 2 + self.num_wan):
            sp = ln[i].split()
            if sp[0] == "X":
                tmp = []
                for j in range(3):
                    tmp.append(float(sp[j + 1]))
                xyz_cen.append(tmp)
            else:
                raise Exception("Inconsistency in the centres file.")
        self.xyz_cen = np.array(xyz_cen, dtype=float)
        # get orbital positions in reduced coordinates
        self.red_cen = _cart_to_red((self.lat[0], self.lat[1], self.lat[2]), self.xyz_cen)

    def model(self, zero_energy=0.0, min_hopping_norm=None, max_distance=None, ignorable_imaginary_part=None):

        # make the model object
        tb = tb_model(3, 3, self.lat, self.red_cen)

        # remember that this model was computed from w90
        tb._assume_position_operator_diagonal = False

        # add onsite energies
        onsite = np.zeros(self.num_wan, dtype=float)
        for i in range(self.num_wan):
            tmp_ham = self.ham_r[(0, 0, 0)]["h"][i, i] / float(self.ham_r[(0, 0, 0)]["deg"])
            onsite[i] = tmp_ham.real
            if np.abs(tmp_ham.imag) > 1.0E-9:
                raise Exception("Onsite terms should be real!")
        tb.set_onsite(onsite - zero_energy)

        # add hopping terms
        for R in self.ham_r:
            # avoid double counting
            use_this_R = True
            # avoid onsite terms
            if R[0] == 0 and R[1] == 0 and R[2] == 0:
                avoid_diagonal = True
            else:
                avoid_diagonal = False
                # avoid taking both R and -R
                if R[0] != 0:
                    if R[0] < 0:
                        use_this_R = False
                else:
                    if R[1] != 0:
                        if R[1] < 0:
                            use_this_R = False
                    else:
                        if R[2] < 0:
                            use_this_R = False
            # get R vector
            vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]), [R])[0]
            # scan through unique R
            if use_this_R:
                for i in range(self.num_wan):
                    vec_i = self.xyz_cen[i]
                    for j in range(self.num_wan):
                        vec_j = self.xyz_cen[j]
                        # get distance between orbitals
                        dist_ijR = np.sqrt(np.dot(-vec_i + vec_j + vecR,
                                                  -vec_i + vec_j + vecR))
                        # to prevent double counting
                        if not (avoid_diagonal == True and j <= i):

                            # only if distance between orbitals is small enough
                            if max_distance is not None:
                                if dist_ijR > max_distance:
                                    continue

                            # divide the matrix element from w90 with the degeneracy
                            tmp_ham = self.ham_r[R]["h"][i, j] / float(self.ham_r[R]["deg"])

                            # only if big enough matrix element
                            if min_hopping_norm is not None:
                                if np.abs(tmp_ham) < min_hopping_norm:
                                    continue

                            # remove imaginary part if needed
                            if ignorable_imaginary_part is not None:
                                if np.abs(tmp_ham.imag) < ignorable_imaginary_part:
                                    tmp_ham = tmp_ham.real + 0.0j

                            # set the hopping term
                            tb.set_hop(tmp_ham, i, j, list(R))

        return tb

    def dist_hop(self):

        ret_ham = []
        ret_dist = []
        for R in self.ham_r:
            # treat diagonal terms differently
            if R[0] == 0 and R[1] == 0 and R[2] == 0:
                avoid_diagonal = True
            else:
                avoid_diagonal = False

            # get R vector
            vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]), [R])[0]
            for i in range(self.num_wan):
                vec_i = self.xyz_cen[i]
                for j in range(self.num_wan):
                    vec_j = self.xyz_cen[j]
                    # diagonal terms
                    if not (avoid_diagonal == True and i == j):
                        # divide the matrix element from w90 with the degeneracy
                        ret_ham.append(self.ham_r[R]["h"][i, j] / float(self.ham_r[R]["deg"]))

                        # get distance between orbitals
                        ret_dist.append(np.sqrt(np.dot(-vec_i + vec_j + vecR, -vec_i + vec_j + vecR)))

        return np.array(ret_dist), np.array(ret_ham)


def _cart_to_red(tmp, cart):
    "Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors"
    (a1,a2,a3) =tmp
    #matrix with lattice vectors
    cnv = np.array([a1, a2, a3])
    #transpose a matrix
    cnv=cnv.T
    #invert a matrix
    cnv=np.linalg.inv(cnv)
    #reduced coordinates
    red = np.zeros_like(cart, dtype=float)
    for i in range(0, len(cart)):
        red[i]=np.dot(cnv, cart[i])
    return red

def _red_to_cart(tmp, red):
    (a1, a2, a3)= tmp
    cart = np.zeros_like(red, dtype=float)
    for i in range(0, len(cart)):
        cart[i,:]=a1*red[i][0]+a2*red[i][1]+a3*red[i][2]
    return cart

class tb_model(object):
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):

        #dimensionality of k-space (integer)
        self._dim_k=dim_k

        #dimensionality of r-space (integer)
        self._dim_r = dim_r

        #lattice vector
        self._lat = np.array(lat, dtype=float)


        #number of basis orbitals per cell
        self._orb= np.array(orb, dtype=float)
        if len(self._orb.shape) !=2:
            raise Exception("\n\nWrong orb array rank")
        self._norb = self._orb.shape[0]
        if self._orb.shape[1] !=dim_r:
            raise Exception("\n\nWrong orb array dimension")

        if per is None:
            # by default first _dim_k dimensions are periodic
            self._per = list(range(self._dim_k))
        else:
            if len(per) != self._dim_k:
                raise Exception("\n\nWrong choice of periodic/infinite direction!")
            # store which directions are the periodic ones
            self._per = per

        # remember number of spin components
        self._nspin = nspin

        # by default, assume model did not come from w90 object and that
        # position operator is diagonal
        self._assume_position_operator_diagonal = True

        # compute number of electronic states at each k-point
        self._nsta = self._norb*self._nspin

        # Initialize onsite energies to zero
        if self._nspin == 1:
            self._site_energies = np.zeros(self._norb, dtype=float)

        # remember which onsite energies user has specified
        self._site_energies_specified = np.zeros(self._norb, dtype=bool)
        self._site_energies_specified[:] = False

        # Initialize hoppings to empty list
        self._hoppings = []

    def set_onsite(self, onsite_en, ind_i=None, mode="set"):
        if ind_i is None:
            if len(onsite_en) !=self._norb:
                raise Exception("\n\nWrong number of site energies")

        if ind_i is not None:
            if ind_i<0 or ind_i>=self._norb:
                raise Exception("\n\nIndex ind_i out of space")

        if ind_i is not None:
            to_check=[onsite_en]
        else:
            to_check= onsite_en
        for ons in to_check:
            if np.array(ons).shape==():
                if np.abs(np.array(ons)-np.array(ons).conjugate())>1.0E-8:
                    raise Exception("\n\nOnsite energy should not have imaginary part!")
            elif np.array(ons).shape==(4,):
                if np.max(np.abs(np.array(ons)-np.array(ons).conjugate())) > 1.0E-8:
                    raise Exception("\n\nOnsite energy or Zeeman field should not have imaginary part!")
            elif np.array(ons).shape==(2,2):
                if np.max(np.abs(np.array(ons)-np.array(ons).T.conjugate()))>1.0E-8:
                    raise Exception("\n\nOnsite matrix should be Hermitian")

            #specific onsite energies from scratch, can be called only once
        if mode.lower() == "set":
             # specifying only one site at a time
            if ind_i is not None:
                # make sure we specify things only once
                if self._site_energies_specified[ind_i]:
                    raise Exception(
                            "\n\nOnsite energy for this site was already specified! Use mode=\"reset\" or mode=\"add\".")
                else:
                        self._site_energies[ind_i] = self._val_to_block(onsite_en)
                        self._site_energies_specified[ind_i] = True
                # specifying all sites at once
            else:
                    # make sure we specify things only once
                if True in self._site_energies_specified[ind_i]:
                    raise Exception(
                            "\n\nSome or all onsite energies were already specified! Use mode=\"reset\" or mode=\"add\".")
                else:
                     for i in range(self._norb):
                        self._site_energies[i] = self._val_to_block(onsite_en[i])
                self._site_energies_specified[:] = True

        else:
            raise Exception("\n\nWrong value of mode parameter")


    def set_hop(self, hop_amp, ind_i, ind_j, ind_R=None, mode="set"):
        if self._dim_k != 0 and (ind_R is None):
            raise Exception("\n\nNeed to specify ind_R!")
        # if necessary convert from integer to array
        if self._dim_k == 1 and type(ind_R).__name__ == 'int':
            tmpR = np.zeros(self._dim_r, dtype=int)
            tmpR[self._per] = ind_R
            ind_R = tmpR
        # check length of ind_R
        if self._dim_k != 0:
            if len(ind_R) != self._dim_r:
                raise Exception("\n\nLength of input ind_R vector must equal dim_r! Even if dim_k<dim_r.")
        # make sure ind_i and ind_j are not out of scope
        if ind_i < 0 or ind_i >= self._norb:
            raise Exception("\n\nIndex ind_i out of scope.")
        if ind_j < 0 or ind_j >= self._norb:
            raise Exception("\n\nIndex ind_j out of scope.")
            # do not allow onsite hoppings to be specified here because then they
        # will be double-counted
        if self._dim_k == 0:
            if ind_i == ind_j:
                raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
        else:
            if ind_i == ind_j:
                all_zer = True
                for k in self._per:
                    if int(ind_R[k]) != 0:
                        all_zer = False
                if all_zer:
                    raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
        #
        # convert to 2by2 matrix if needed
        hop_use = self._val_to_block(hop_amp)
        # hopping term parameters to be stored
        if self._dim_k == 0:
            new_hop = [hop_use, int(ind_i), int(ind_j)]
        else:
            new_hop = [hop_use, int(ind_i), int(ind_j), np.array(ind_R)]
        #
        # see if there is a hopping term with same i,j,R
        use_index = None
        for iih, h in enumerate(self._hoppings):
            # check if the same
            same_ijR = False
            if ind_i == h[1] and ind_j == h[2]:
                if self._dim_k == 0:
                    same_ijR = True
                else:
                    if False not in (np.array(ind_R)[self._per] == np.array(h[3])[self._per]):
                        same_ijR = True
            # if they are the same then store index of site at which they are the same
            if same_ijR:
                use_index = iih
        #
        # specifying hopping terms from scratch, can be called only once
        if mode.lower() == "set":
            # make sure we specify things only once
            if use_index is not None:
                raise Exception(
                    "\n\nHopping energy for this site was already specified! Use mode=\"reset\" or mode=\"add\".")
            else:
                self._hoppings.append(new_hop)

    def _val_to_block(self, val):
        if self._nspin==1:
            return val
        elif self._nspin==2:
            ret = np.zeros((2,2), dtype=complex)

            use_val = np.array(val)
            if use_val.shape()==():
                ret[0,0]+=use_val
                ret[1,1]+=use_val
            elif use_val.shape == (4,):
                # diagonal
                ret[0, 0] += use_val[0]
                ret[1, 1] += use_val[0]
                # sigma_x
                ret[0, 1] += use_val[1]
                ret[1, 0] += use_val[1]  # sigma_y
                ret[0, 1] += use_val[2] * (-1.0j)
                ret[1, 0] += use_val[2] * (1.0j)
                # sigma_z
                ret[0, 0] += use_val[3]
                ret[1, 1] += use_val[3] * (-1.0)
                # if 2 by 2 matrix is given
            elif use_val.shape == (2, 2):
                return use_val
            else:
                raise Exception("Wrong format")

    def _gen_ham(self, k_input=None):
        """Generate Hamiltonian for a certain k-point, K-point is given in reduced coordinates!"""
        kpnt = np.array(k_input)
        if not (k_input is None):
            # if kpnt is just a number then convert it to an array
            if len(kpnt.shape) == 0:
                kpnt = np.array([kpnt])
            # check that k-vector is of corect size
            if kpnt.shape != (self._dim_k,):
                raise Exception("\n\nk-vector of wrong shape!")
        else:
            if self._dim_k != 0:
                raise Exception("\n\nHave to provide a k-vector!")
        # zero the Hamiltonian matrix
        if self._nspin == 1:
            ham = np.zeros((self._norb, self._norb), dtype=complex)
        elif self._nspin == 2:
            ham = np.zeros((self._norb, 2, self._norb, 2), dtype=complex)
        # modify diagonal elements
        for i in range(self._norb):
            if self._nspin == 1:
                ham[i, i] = self._site_energies[i]
            elif self._nspin == 2:
                ham[i, :, i, :] = self._site_energies[i]
        # go over all hoppings
        for hopping in self._hoppings:
            # get all data for the hopping parameter
            if self._nspin == 1:
                amp = complex(hopping[0])
            elif self._nspin == 2:
                amp = np.array(hopping[0], dtype=complex)
            i = hopping[1]
            j = hopping[2]
            # in 0-dim case there is no phase factor
            if self._dim_k > 0:
                ind_R = np.array(hopping[3], dtype=float)
                # vector from one site to another
                rv = -self._orb[i, :] + self._orb[j, :] + ind_R
                # Take only components of vector which are periodic
                rv = rv[self._per]
                # Calculate the hopping, see details in info/tb/tb.pdf
                phase = np.exp(2.0j * np.pi * np.dot(kpnt, rv))
                amp = amp * phase
            # add this hopping into a matrix and also its conjugate
            if self._nspin == 1:
                ham[i, j] += amp
                ham[j, i] += amp.conjugate()
            elif self._nspin == 2:
                ham[i, :, j, :] += amp
                ham[j, :, i, :] += amp.T.conjugate()
        return ham

    def _sol_ham(self,ham,eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        if self._nspin==1:
            ham_use=ham
        elif self._nspin==2:
            ham_use=ham.reshape((2*self._norb,2*self._norb))
        # check that matrix is hermitian
        if np.max(ham_use-ham_use.T.conj())>1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix
        if not eig_vectors: # only find eigenvalues
            eval=np.linalg.eigvalsh(ham_use)
            # sort eigenvalues and convert to real numbers
            eval=_nicefy_eig(eval)
            return np.array(eval,dtype=float)
        else: # find eigenvalues and eigenvectors
            (eval,eig)=np.linalg.eigh(ham_use)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig=eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval,eig)=_nicefy_eig(eval,eig)
            # reshape eigenvectors if doing a spinfull calculation
            if self._nspin==2:
                eig=eig.reshape((self._nsta,self._norb,2))
            return (eval,eig)

    def solve_all(self, k_list=None, eig_vectors=False):

        if not (k_list is None):
            nkp=len(k_list)  #number of k_points

            ret_eval = np.zeros((self._nsta, nkp), dtype=float)

            if self._nspin==1:
                ret_evec=np.zeros((self._nsta,nkp,self._norb),dtype=complex)
            elif self._nspin==2:
                ret_evec = np.zeros((self._nsta, nkp, self._norb,2), dtype=complex)
            for i,k in enumerate(k_list):
                ham=self._gen_ham(k)
                #solve Hamiltonian
                if not eig_vectors:
                    eval=self._sol_ham(ham, eig_vectors=eig_vectors)
                    ret_eval[:,i]=eval[:]
                else:
                    (eval, evec)=self._sol_ham(ham, eig_vectors=eig_vectors)
                    ret_eval[:,i]=eval[:]
                    if self._nspin==1:
                        ret_evec[:,i,:]=evec[:,:]
                    elif self._nspin==2:
                        ret_evec[:,i,:,:]=evec[:,:,:]
            if not eig_vectors:
                return ret_eval
            else:
                return ret_eval, ret_evec
        else: # 0 dim case
            # generate Hamiltonian
            ham=self._gen_ham()
            # solve
            if not eig_vectors:
                eval=self._sol_ham(ham,eig_vectors=eig_vectors)
                # indices of eval are [band]
                return eval
            else:
                (eval,evec)=self._sol_ham(ham,eig_vectors=eig_vectors)
                # indices of eval are [band] and of evec are [band,orbital,spin]
                return eval, evec

    def k_path(self, kpts, nk, report=True):

        k_list=np.array(kpts)

        # in 1D case if path is specified as a vector, convert it to an (n,1) array
        if len(k_list.shape) == 1 and self._dim_k == 1:
            k_list = np.array([k_list]).T

        # make sure that k-points in the path have correct dimension
        if k_list.shape[1] != self._dim_k:
            print('input k-space dimension is', k_list.shape[1])
            print('k-space dimension taken from model is', self._dim_k)
            raise Exception("\n\nk-space dimensions do not match")

        # must have more k-points in the path than number of nodes
        if nk < k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes = k_list.shape[0]

        # extract the lattice vectors from the TB model
        lat_per = np.copy(self._lat)
        # choose only those that correspond to periodic directions
        lat_per = lat_per[self._per]
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per, lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node = np.zeros(n_nodes, dtype=float)
        for n in range(1, n_nodes):
            dk = k_list[n] - k_list[n - 1]
            dklen = np.sqrt(np.dot(dk, np.dot(k_metric, dk)))
            k_node[n] = k_node[n - 1] + dklen

        # Find indices of nodes in interpolated list
        node_index = [0]
        for n in range(1, n_nodes - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (nk - 1))))
        node_index.append(nk - 1)

        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist = np.zeros(nk, dtype=float)
        #   array listing the interpolated k-points
        k_vec = np.zeros((nk, self._dim_k), dtype=float)

        # go over all kpoints
        k_vec[0] = k_list[0]
        for n in range(1, n_nodes):
            n_i = node_index[n - 1]
            n_f = node_index[n]
            kd_i = k_node[n - 1]
            kd_f = k_node[n]
            k_i = k_list[n - 1]
            k_f = k_list[n]
            for j in range(n_i, n_f + 1):
                frac = float(j - n_i) / float(n_f - n_i)
                k_dist[j] = kd_i + frac * (kd_f - kd_i)
                k_vec[j] = k_i + frac * (k_f - k_i)

        if report:
            if self._dim_k == 1:
                print(' Path in 1D BZ defined by nodes at ' + str(k_list.flatten()))
            else:
                print('----- k_path report begin ----------')
                original = np.get_printoptions()
                np.set_printoptions(precision=5)
                print('real-space lattice vectors\n', lat_per)
                print('k-space metric tensor\n', k_metric)
                print('internal coordinates of nodes\n', k_list)
                if lat_per.shape[0] == lat_per.shape[1]:
                    # lat_per is invertible
                    lat_per_inv = np.linalg.inv(lat_per).T
                    print('reciprocal-space lattice vectors\n', lat_per_inv)
                    # cartesian coordinates of nodes
                    kpts_cart = np.tensordot(k_list, lat_per_inv, axes=1)
                    print('cartesian coordinates of nodes\n', kpts_cart)
                print('list of segments:')
                for n in range(1, n_nodes):
                    dk = k_node[n] - k_node[n - 1]
                    dk_str = _nice_float(dk, 7, 5)
                    print('  length = ' + dk_str + '  from ', k_list[n - 1], ' to ', k_list[n])
                print('node distance list:', k_node)
                print('node index list:   ', np.array(node_index))
                np.set_printoptions(precision=original["precision"])
                print('----- k_path report end ------------')
            print()

        return k_vec, k_dist, k_node

    def dist_hop(self):
        r"""
        This is one of the diagnostic tools that can be used to help
        in determining *min_hopping_norm* and *max_distance* parameter in
        :func:`pythtb.w90.model` function call."""

        ret_ham = []
        ret_dist = []
        for R in self.ham_r:
            # treat diagonal terms differently
            if R[0] == 0 and R[1] == 0 and R[2] == 0:
                avoid_diagonal = True
            else:
                avoid_diagonal = False

            # get R vector
            vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]), [R])[0]
            for i in range(self.num_wan):
                vec_i = self.xyz_cen[i]
                for j in range(self.num_wan):
                    vec_j = self.xyz_cen[j]
                    # diagonal terms
                    if not (avoid_diagonal == True and i == j):
                        # divide the matrix element from w90 with the degeneracy
                        ret_ham.append(self.ham_r[R]["h"][i, j] / float(self.ham_r[R]["deg"]))

                        # get distance between orbitals
                        ret_dist.append(np.sqrt(np.dot(-vec_i + vec_j + vecR, -vec_i + vec_j + vecR)))

        return np.array(ret_dist), np.array(ret_ham)

    def display(self):
        r"""
        Prints on the screen some information about this tight-binding
        model. This function doesn't take any parameters.
        """
        print('---------------------------------------')
        print('report of tight-binding model')
        print('---------------------------------------')
        print('k-space dimension           =',self._dim_k)
        print('r-space dimension           =',self._dim_r)
        print('number of spin components   =',self._nspin)
        print('periodic directions         =',self._per)
        print('number of orbitals          =',self._norb)
        print('number of electronic states =',self._nsta)
        print('lattice vectors:')
        for i,o in enumerate(self._lat):
            print(" #",_nice_int(i,2)," ===>  [", end=' ')
            for j,v in enumerate(o):
                print(_nice_float(v,7,4), end=' ')
                if j!=len(o)-1:
                    print(",", end=' ')
            print("]")
        print('positions of orbitals:')
        for i,o in enumerate(self._orb):
            print(" #",_nice_int(i,2)," ===>  [", end=' ')
            for j,v in enumerate(o):
                print(_nice_float(v,7,4), end=' ')
                if j!=len(o)-1:
                    print(",", end=' ')
            print("]")
        print('site energies:')
        for i,site in enumerate(self._site_energies):
            print(" #",_nice_int(i,2)," ===>  ", end=' ')
            if self._nspin==1:
                print(_nice_float(site,7,4))
            elif self._nspin==2:
                print(str(site).replace("\n"," "))
        print('hoppings:')
        for i,hopping in enumerate(self._hoppings):
            print("<",_nice_int(hopping[1],2),"| H |",_nice_int(hopping[2],2), end=' ')
            if len(hopping)==4:
                print("+ [", end=' ')
                for j,v in enumerate(hopping[3]):
                    print(_nice_int(v,2), end=' ')
                    if j!=len(hopping[3])-1:
                        print(",", end=' ')
                    else:
                        print("]", end=' ')
            print(">     ===> ", end=' ')
            if self._nspin==1:
                print(_nice_complex(hopping[0],7,4))
            elif self._nspin==2:
                print(str(hopping[0]).replace("\n"," "))
        print('hopping distances:')
        for i,hopping in enumerate(self._hoppings):
            print("|  pos(",_nice_int(hopping[1],2),")  - pos(",_nice_int(hopping[2],2), end=' ')
            if len(hopping)==4:
                print("+ [", end=' ')
                for j,v in enumerate(hopping[3]):
                    print(_nice_int(v,2), end=' ')
                    if j!=len(hopping[3])-1:
                        print(",", end=' ')
                    else:
                        print("]", end=' ')
            print(") |  =  ", end=' ')
            pos_i=np.dot(self._orb[hopping[1]],self._lat)
            pos_j=np.dot(self._orb[hopping[2]],self._lat)
            if len(hopping)==4:
                pos_j+=np.dot(hopping[3],self._lat)
            dist=np.linalg.norm(pos_j-pos_i)
            print (_nice_float(dist,7,4))

        print()



def _nicefy_eig(eval,eig=None):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=np.array(eval.real,dtype=float)
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    if not (eig is None):
        eig=eig[args]
        return eval, eig
    return eval

# for nice justified printout
def _nice_float(x,just,rnd):
    return str(round(x,rnd)).rjust(just)
def _nice_int(x,just):
    return str(x).rjust(just)
def _nice_complex(x,just,rnd):
    ret=""
    ret+=_nice_float(complex(x).real,just,rnd)
    if complex(x).imag<0.0:
        ret+=" - "
    else:
        ret+=" + "
    ret+=_nice_float(abs(complex(x).imag),just,rnd)
    ret+=" i"
    return ret









