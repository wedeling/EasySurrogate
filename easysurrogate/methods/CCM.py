import numpy as np
from scipy import stats
from itertools import chain, product, cycle
import matplotlib.pyplot as plt
import sys
from scipy.spatial import ConvexHull

class CCM:
    
    def __init__(self, c, r, N_bins, lags, min_count = 0):
        
        #spatial dimension, hardcoded to 1 for now
        self.N = 1
        
        #total number of conditional variables including lagged terms
        self.N_c = c.shape[1]
        self.N_s = c.shape[0]
        
        #number of unique conditioning variables (not including lags)
        self.N_covar = len(lags)

        self.r = r
        self.c = c
        self.N_bins = N_bins
        self.lags = lags
        self.max_lag = np.max(list(chain(*lags)))
        self.covar = {}
        
        for i in range(self.N_covar):
            self.covar[i] = []

        bins = self.get_bins(N_bins)
        
        count, binedges, binnumber = stats.binned_statistic_dd(c, np.zeros(self.N_s), 
                                                               statistic='count', bins=bins)
        
        #the unique set of binnumers which have at least one r_ip1 sample
        unique_binnumbers = np.unique(binnumber)
        
        #some scalars
        binnumber_max = np.max(unique_binnumbers)
        
        #array containing r_ip1 indices sorted PER BIN
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #SHOULD BE A 1D ARRAY, AN ARRAY OF SIZE [MAX(BINNUMBER), MAX(COUNT)]
        #WILL STORE MOSTLY ZEROS IN HIGHER DIMENSIONS, LEADING TO MEMORY FAILURES
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        idx_of_bin = []
        
        #number of samples in each bin
        size_of_bin = []
        
        for b in unique_binnumbers:
            #r_ip1 indices of bin b
            tmp = np.where(binnumber == b)[0]
            
            idx_of_bin.append(tmp)
            size_of_bin.append(tmp.size)

        #make a 1d array of the nested arrays
        idx_of_bin = np.array(list(chain(*idx_of_bin)))
        size_of_bin = np.array(size_of_bin)
        
        #the starting offset of each bin in the 1D array idx_of_bin
        #offset[unique_binnumbers] will give the starting index in idex_of_bin for
        #each bin. Some entries are zero, which correspond to empty bins (except for
        #the lowest non-empty binnumber)
        offset = np.zeros(unique_binnumbers.size).astype('int')
        offset[1:] = np.cumsum(size_of_bin[0:-1])
        tmp = np.zeros(binnumber_max+1).astype('int')
        tmp[unique_binnumbers] = offset
        self.offset = tmp
        
        ####################
        #empty bin handling#
        ####################
        
        #find indices of empty bins
        #Note: if 'full' is defined as 1 or more sample, binnumbers_nonempty is 
        #the same as binnumbers_unique
        x_idx_nonempty = np.where(count > min_count)
        x_idx_nonempty_p1 = [x_idx_nonempty[i] + 1 for i in range(self.N_c)]
        binnumbers_nonempty = np.ravel_multi_index(x_idx_nonempty_p1, [len(b) + 1 for b in bins]) 
        N_nonempty = binnumbers_nonempty.size
        
        #mid points of the bins
        x_mid = [0.5*(bins[i][1:] + bins[i][0:-1]) for i in range(self.N_c)]

        #mid points of the non-empty bins
        midpoints = np.zeros([N_nonempty, self.N_c])
        for i in range(self.N_c):
            midpoints[:, i] = x_mid[i][x_idx_nonempty[i]]
            
        #########################################
        # find all sample indices per bin index #
        #########################################
        
        sample_idx_per_bin = {}
        for j in unique_binnumbers:
            sample_idx_per_bin[j] = np.where(binnumber == j)[0]

        self.bins = bins
        self.count = count
        self.binnumber = binnumber
        self.binnumbers_nonempty = binnumbers_nonempty
        self.midpoints = midpoints
        self.idx_of_bin = idx_of_bin
        self.unique_binnumbers = unique_binnumbers
        self.sample_idx_per_bin = sample_idx_per_bin
        
        print('Connecting all possible empty bins to nearest non-empty bin...')
        self.fill_in_blanks()
        print('done.')


    #check which c_i fall within empty bins and correct binnumbers_i by
    #projecting to the nearest non-empty bin
    def check_outliers(self, binnumbers_i, c_i):
        
        #find out how many BINS with outliers there are
        unique_binnumbers_i = np.unique(binnumbers_i)
        idx = np.where(np.in1d(unique_binnumbers_i, self.binnumbers_nonempty) == False)[0]
        N_outlier_bins = idx.size
        
        if N_outlier_bins > 0:
            
            #index of outlier SAMPLES in binnumbers_i
            outliers_idx = np.in1d(binnumbers_i, unique_binnumbers_i[idx]).nonzero()[0]
            N_outliers = outliers_idx.size
            
#            if self.verbose == True:
#                print(N_outlier_bins, ' bins with', N_outliers ,'outlier samples found')
        
            #x location of outliers
            x_outliers = np.copy(c_i[outliers_idx])
        
            #find non-empty bin closest to the outliers
            closest_idx = np.zeros(N_outliers).astype('int')
            for i in range(N_outliers):
                if self.N_c == 1:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i], 2, 1)
                else:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i,:], 2, 1)
                closest_idx[i] = np.argmin(dist)
             
            binnumbers_closest = self.binnumbers_nonempty[closest_idx]
            
            check = self.binnumbers_nonempty[closest_idx]
            x_idx_check = np.unravel_index(check, [len(b) + 1 for b in self.bins])
            x_idx_check = [x_idx_check[i] - 1 for i in range(self.N_c)]
            
            #overwrite outliers in binnumbers_i with nearest non-empty binnumber
            binnumbers_closest = self.binnumbers_nonempty[closest_idx]

#            if self.verbose == True:
#                print('Moving', binnumbers_i[outliers_idx], '-->', binnumbers_closest)

            binnumbers_i[outliers_idx] = binnumbers_closest
            
    def fill_in_blanks(self):
        """
        Create an a-priori mapping between every possible bin and the nearest 
        non-empty bin. Non-empty bins will link to themselves.
    
        Returns
        -------
        None.

        """
        
        bins_padded = []
        
        #mid points of all possible bins, including ghost bins
        for i in range(self.N_c):
            dx1 = self.bins[i][1] - self.bins[i][0]
            dx2 = self.bins[i][-1] - self.bins[i][-2]
            
            #pad the beginning and end of current 1D bin with extrapolated values
            bin_pad = np.pad(self.bins[i], (1,1), 'constant', constant_values=(self.bins[i][0]-dx1, self.bins[i][-1]+dx2))    
            bins_padded.append(bin_pad)
        
        #compute the midpoints of the padded bins
        x_mid_pad = [0.5*(bins_padded[i][1:] + bins_padded[i][0:-1]) for i in range(self.N_c)]
        self.x_mid_pad_tensor = np.array(list(product(*x_mid_pad)))
        
        #total number bins
        self.max_binnumber = self.x_mid_pad_tensor.shape[0]
        
        mapping = np.zeros(self.max_binnumber).astype('int')
        
        for i in range(self.max_binnumber):
            
            if np.mod(i, 10) == 0:
                progress(i, self.max_binnumber)
            
            #bin is nonempty, just use current idx
            # if np.in1d(i, self.unique_binnumbers) == True:
            if np.in1d(i, self.binnumbers_nonempty) == True:
                mapping[i] = i
            #bin is empty, find nearest non-empty bin
            else:
                binnumbers_i = np.array([i])
                self.check_outliers(binnumbers_i, self.x_mid_pad_tensor[i].reshape([1, self.N_c]))
                mapping[i] = binnumbers_i[0]
            
        self.mapping = mapping
        

    def plot_2D_binning_object(self):
        """
        Visual representation of a 2D binning object. Also shows the mapping
        between empty to nearest non-empty bins.

        Returns
        -------
        None.

        """
        if self.N_c != 2:
            print('Only works for N_c = 2')
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=r'conditioning variable 1', ylabel=r'conditioning variable 2')
        
        #plot bins and (c1, c2) which corresponding to a r sample point
        ax.plot(self.c[:,0], self.c[:,1], '+', color='lightgray', alpha=0.3)
        ax.vlines(self.bins[0], np.min(self.c[:,1]), np.max(self.c[:,1]))
        ax.hlines(self.bins[1], np.min(self.c[:,0]), np.max(self.c[:,0]))
       
        ax.plot(self.x_mid_pad_tensor[:,0], self.x_mid_pad_tensor[:,1], 'g+')

        #plot the mapping
        for i in range(self.max_binnumber):
            ax.plot([self.x_mid_pad_tensor[i][0], self.x_mid_pad_tensor[self.mapping[i]][0]], \
                    [self.x_mid_pad_tensor[i][1], self.x_mid_pad_tensor[self.mapping[i]][1]], 'b', alpha=0.6)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.show()
        
    def onclick(self, event):
        
        if event.inaxes != self.ax1: return
        
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        
        c_i = np.array([event.xdata, event.ydata]).reshape([1, 2])
        _, _, binnumber_i = stats.binned_statistic_dd(c_i, np.zeros(1), 
                                                      bins=self.bins)

        print('Bin', binnumber_i[0])
        binnumber_i = self.mapping[binnumber_i][0]
        idx = self.sample_idx_per_bin[binnumber_i]
        points_c = self.c[idx]
        points_r = self.r[idx]
                    
        self.plot_local_bin(points_c, binnumber_i, self.ax1, 'r', width = 4)               
        self.plot_local_bin(points_r, binnumber_i, self.ax2, 'r', width = 4)
        
        plt.draw()
        
    
    def plot_local_bin(self, points, binnumber, ax, marker, width=2):
        #if  there are 3 or more samples in current bin, plot the
        #convex hull of all samples in this bin
        if points.shape[0] >= 3:        
            # marker = next(colors)
            hull = ConvexHull(points)
            ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], marker, linewidth=width)
            ax.plot([points[hull.vertices[0], 0], points[hull.vertices[-1], 0]],
                    [points[hull.vertices[0], 1], points[hull.vertices[-1], 1]], marker, linewidth=width)
            print(hull.volume)
        #     #plot the binnumber
        #     x_mid = np.mean(points[hull.vertices, :], axis=0)
        #     ax.text(x_mid[0], x_mid[1], str(binnumber))
        # # ax.plot(X[idx_i, 0], X[idx_i, 1], '+')
        # else:
        #     x_mid = np.mean(points, axis=0)
        #     ax.text(x_mid[0], x_mid[1], str(binnumber))
        
        
    def plot_2D_shadow_manifold(self):
        """
        Plots the bins on the manifold of the conditioning variables, and
        the corresponding neighborhoods on the shadow manifold.   

        Returns
        -------
        None.

        """

        if self.N_c == 1:
            print('Dimension manifold = 1, only works for dimension of 2 or higher')
            return
        elif self.N_c > 2:
            print('Dimension manifold > 2, will only plot first 2 dimensions')

        fig = plt.figure('manifolds_and_binnumbers', figsize = [8, 4])
        # colors = cycle(['--r', '--g', '--b', '--m', '--k'])
        self.ax1 = fig.add_subplot(121, title = 'Manifold')
        self.ax2 = fig.add_subplot(122, title = 'Shawdow manifold')       
        self.ax1.set_xlabel(r'$\mathrm{conditioning\;variable\;at\;t-\tau}$')
        self.ax1.set_ylabel(r'$\mathrm{conditioning\;variable\;at\;t}$')
        self.ax2.set_xlabel(r'$\mathrm{shadow\;variable\;at\;t-\tau}$')
        self.ax2.set_ylabel(r'$\mathrm{shadow\;variable\;at\;t}$')
        
        #plot the samples
        self.ax1.plot(self.c[:,0], self.c[:,1], '+', color='lightgray', alpha=0.3)
        self.ax2.plot(self.r[:,0], self.r[:,1], '+', color='lightgray', alpha=0.3)
        
        for idx_i in self.sample_idx_per_bin.keys():
            
            idx = self.sample_idx_per_bin[idx_i]
            points_c = self.c[idx, 0:2]
            points_r = self.r[idx, 0:2]
                        
            self.plot_local_bin(points_c, idx_i, self.ax1, '--k')               
            self.plot_local_bin(points_r, idx_i, self.ax2, '--k')               

        plt.tight_layout()
        
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

    def compare_convex_hull_volumes(self):
        
        self.vols = []
        self.ratio_vols = []
        
        total_vol_c = ConvexHull(self.c).volume
        total_vol_r = ConvexHull(self.r).volume
        
        weights = []
        
        for idx_i in self.sample_idx_per_bin.keys():

            idx = self.sample_idx_per_bin[idx_i]
            points_c = self.c[idx]
            points_r = self.r[idx]     

            if points_c.shape[0] >= self.N_c + 1:

                hull_c = ConvexHull(points_c)
                hull_r = ConvexHull(points_r)
                
                vol_fraction_c = hull_c.volume/total_vol_c
                vol_fraction_r = hull_r.volume/total_vol_r
                
                weights.append(vol_fraction_c)
               
                self.vols.append([vol_fraction_c, vol_fraction_r])
                self.ratio_vols.append(vol_fraction_c/vol_fraction_r)
                
        avg_ratio = np.mean(self.ratio_vols)
        
        weights = weights/np.sum(weights)    
        print(weights)
        print(np.sum(weights))
        weighted_ratio = np.mean(weights*self.ratio_vols)
        self.weights = weights
        
        print('Average volume ratio binning cell/shadow cell = %.3f' % avg_ratio)
        print('Max volume ratio binning cell/shadow cell = %.3f' % np.max(self.ratio_vols))
        print('min volume ratio binning cell/shadow cell = %.3f' % np.min(self.ratio_vols))
        
        print('Weighted average volume ratio binning cell/shadow cell = %.3f' % weighted_ratio)

    def get_sample(self, c_i, stochastic = False):
    
        #find in which bins the c_i samples fall
        _, _, binnumbers_i = stats.binned_statistic_dd(c_i, np.zeros(self.N**2), bins=self.bins)
        
        #static correction for outliers, using precomputed mapping array
        binnumbers_i = self.mapping[binnumbers_i]
       
        #the neighbors in the selected bin, on the 'C manifold'
        neighbors = self.c[self.sample_idx_per_bin[binnumbers_i[0]]]
        
        #the distance from the current point to all neighbors
        dists = np.linalg.norm(neighbors - c_i, axis = 1)
        
        #if the current bin does not contain enough points to form a simplex,
        #adjust K.
        if dists.size < self.N_c + 1:
            K = dists.size      #adjusted K
        else:
            K = self.N_c + 1    #simplex K (manifold dimension + 1)
        
        #sort the distances + select the K nearest neighbors
        idx = np.argsort(dists)[0:K]
        simplex_idx = self.sample_idx_per_bin[binnumbers_i[0]][idx]
        
        if not stochastic:
            #compute the simplex weights w_i
            d_i = dists[idx]
            if d_i[0] == 0:
                u_i = np.zeros(K)
                u_i[0] = 1.0
            else:
                u_i = np.exp(-d_i/d_i[0])
            w_i = u_i/np.sum(u_i)
    
            #prediction is the weighted sum of correspoding samples
            #on the showdow manifold        
            shadow_sample = np.sum(self.r[simplex_idx, -1]*w_i)
    
            return shadow_sample
        else:
            shadow_sample = self.sample_simplex(self.r[simplex_idx])[0]
            return shadow_sample[-1]

    def sample_simplex(self, xi, n_mc = 1):
        """
        Use an analytical function map to points in the n_xi-dim. hypercube to a
        n_xi-dim simplex with vertices xi_k_jl. 
        
        Source: 
            W. Edeling, R. Dwight, P. Cinnella, 
            Simplex-stochastic collocation method with improved scalability
            Journal of Computational Physics, 310, 301--328, 2016.
            
        Parameters
        ----------
        xi: array of floats, shape (n_xi + 1, n_xi) 
            
        n_mc : int, number of samples to draw from inside the simplex
            DESCRIPTION. The default is 1.

        Returns
        -------
        n_mc uniformly distributed points inside the simplex.
 
        """
        n_xi = xi.shape[1]
        P = np.zeros([n_mc, n_xi])
        for k in range(n_mc):
            #random points inside the hypercube
            r = np.random.rand(n_xi)
            
            #the term of the map is \xi_k_j0
            sample = np.copy(xi[0])
            for i in range(1, n_xi+1):
                prod_r = 1.
                #compute the product of r-terms: prod(r_{n_xi-j+1}^{1/(n_xi-j+1)})
                for j in range(1,i+1):
                    prod_r *= r[n_xi - j]**(1./(n_xi - j + 1))
                #compute the ith term of the sum: prod_r*(\xi_i-\xi_{i-1})
                sample += prod_r*(xi[i] - xi[i-1])
            P[k,:] = sample
     
        return P
           
    def print_bin_info(self):
        print('-------------------------------')
        print('Total number of samples= ', self.r_ip1.size)
        # print('Total number of bins = ', self.N_bins**self.N_c)
        print('Total number of non-empty bins = ', self.binnumbers_nonempty.size)
        # print('Percentage filled = ', np.double(self.binnumbers_nonempty.size)/self.N_bins**self.N_c*100., ' %')
        print('-------------------------------')
        
    #compute the uniform bins of the conditional variables in c
    def get_bins(self, N_bins):
        
        bins = []
        
        for i in range(self.N_c):
            bins.append(np.linspace(np.min(self.c[:,i]), np.max(self.c[:,i]), N_bins[i]+1))
    
        return bins
    
def progress(count, total, status=''):
    """
    progress bar for the command line
    Source: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
