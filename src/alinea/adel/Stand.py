"""
Class interface for stand generation
"""

import numpy as np
from math import sqrt
from random import sample, random, randrange 
from scipy.interpolate import interp1d

from alinea.adel.stand.stand import agronomicplot, regular_plot


class AgronomicStand(object):
    def __init__(self, sowing_density=10, plant_density=10, inter_row=0.8,
                 noise=0, density_curve_data=None):
        self.sowing_density = sowing_density
        self.inter_row = inter_row
        self.plant_density = plant_density
        self.inter_plant = 1. / inter_row / sowing_density
        self.noise = noise
        self.density_curve_data = density_curve_data
        df = density_curve_data
        if df is None:
            self.density_curve = None
        else:
            # hs_curve = interp1d(df['HS'], df['density'])
            TT_curve = interp1d(df['TT'], df['density'])
            # self.density_curve = {'hs_curve':hs_curve,'TT_curve':TT_curve}
            self.density_curve = TT_curve

    def plot_dimensions(self, nplants=1, aspect='square'):

        if aspect == 'square':
            nsown = nplants * self.sowing_density / float(
                self.plant_density)
            side = sqrt(1. / self.sowing_density * nsown)
            nrow = max(1, round(side / self.inter_row))
            plant_per_row = max(1, round(side / self.inter_plant))
            plot_length = self.inter_plant * plant_per_row
            plot_width = self.inter_row * nrow
            return plot_length, plot_width
        elif aspect == 'line':
            plot_width = self.inter_row
            plot_length = nplants * self.inter_plant * self.sowing_density / float(
                self.plant_density) if self.plant_density > 0. else 0.
            return plot_length, plot_width
        else:
            aspect = float(aspect)
            nsown = nplants * self.sowing_density / float(
                self.plant_density)
            w = sqrt(1. / self.sowing_density * nsown / aspect)
            l = w * aspect
            nrow = max(1, round(w / self.inter_row))
            plant_per_row = max(1, round(l / self.inter_plant))
            plot_length = self.inter_plant * plant_per_row
            plot_width = self.inter_row * nrow
            return plot_length, plot_width

    def smart_stand(self, nplants=1, at=None, convunit=100):
        """ return an (almost) square stand that match inter-row, current density and nplants in the stand, 
             but (dynamicaly) adjusting inter-plant to solve the problem
        """

        density = self.plant_density
        if at is not None:
            if self.density_curve is not None:
                density = self.density_curve(at)

        # find a square design for sowing
        nsown = nplants * 1. * self.sowing_density / density
        side = sqrt(1. / self.sowing_density * nsown)
        nrow = int(max(1, round(side / self.inter_row)))
        plant_per_row = int(max(1, round(float(nsown) / nrow)))
        while nplants > (nrow * plant_per_row):
            plant_per_row += 1
        domain_area = nrow * self.inter_row * plant_per_row * self.inter_plant
        # adjust inter_plant spacing so that n_emerged / domain_area match plant density    
        n_emerged = int(round(domain_area * density))
        # assert(n_emerged >= nplants)
        n_emerged = nplants
        target_domain_area = 1. * n_emerged / density
        inter_plant = target_domain_area / (
        plant_per_row * nrow * self.inter_row)

        positions, domain, domain_area = regular_plot(inter_plant,
                                                      self.inter_row, nrow,
                                                      plant_per_row,
                                                      noise=self.noise,
                                                      convunit=convunit)

        positions = sample(positions, nplants)
        return nplants, domain, positions, domain_area

    def stand(self, nplants=1, aspect='square', convunit=100):

        length, width = self.plot_dimensions(nplants, aspect)
        n_emerged, positions, domain, domain_area, _ = agronomicplot(length,
                                                                     width,
                                                                     self.sowing_density,
                                                                     self.plant_density,
                                                                     self.inter_row,
                                                                     noise=self.noise,
                                                                     convunit=convunit)

        return n_emerged, domain, positions, length * width

    def plot(self, positions):
        import pandas

        df = pandas.DataFrame(positions)
        df.plot(0, 1, style='o')

class FixedSizeStand(object):

    def __init__(self, length, width, n_rows=1, row_shift=0, sowing_density=400,
                 noise=0, positioning='regular', clumping_factor=0, regularity=3):
        '''
        parameter:
            length: The size of the stand in the direction of the lines (in m)
            width: The size of the stand perpendicular to the lines (in m)
            n_rows: The number of rows that will be sown
            row_shift: The distance the rows are shifted from the center towards
                either side. Negative values are possible and change the direction.
            sowing_density: Unused but kept for compatibility with AgronomicStand
            noise: The stddev of the gaussian noise added to the positions
            positioning: 'regular' or 'uniform' that defines how the positioning
                is done.
                'uniform': Each plant position is drawn uniformly from
                the (continuous range of) possible positions. This can lead to
                overlaps and the plants are not equally spaced.
                'regular': The regularity parameter defines how equal the spacing is.
            clumping_factor: The fraction of plants that are not placed according
                to 'positioning' but rather very close to another plant. This
                simulates the clumping of seeds.
            regularity: Parameter used when positioning == 'regular'
        '''
        self.length = length
        self.width = width
        self.plot_dimensions = (length, width)
        self.n_rows = n_rows
        self.row_shift = row_shift
        self.sowing_density = sowing_density
        self.plant_density = sowing_density
        self.noise = noise
        if positioning not in ('regular', 'uniform'):
            raise ValueError("positioning must be either 'regular' or 'uniform'")
        self.positioning = positioning
        self.clumping_factor = clumping_factor
        self.regularity = regularity
        self.density_curve_data = None

    def stand(self, n_plants=1, convunit=100):
        '''
        returns:
            n_plants: The number of plants on this plot
            domain: Pair of points, each point (in convunit) representing a corner of the stand
            positions: The positions where the plants grow, len(positions) == n_plants, list of triplets
            domain_area: The domain area (in m^2)
        '''

        length = self.length * convunit
        width  = self.width  * convunit
        row_shift = self.row_shift * convunit
        n_rows = self.n_rows
        noise = self.noise * convunit
        regularity = 2

        # Define the row positions
        inter_row = width / n_rows
        self.inter_row = inter_row
        row_x = np.linspace(0 + inter_row/2, width - inter_row/2, n_rows)
        row_y = np.ones((n_rows, 2)) * np.array([0 + abs(row_shift)/2, length - abs(row_shift)/2])
        row_y[0::2, :] += row_shift/2
        row_y[1::2, :] -= row_shift/2

        positions = np.zeros((n_plants, 2))
        if self.positioning == 'uniform':
            # Use something similar to a chinese restaurant process to distribute the plants
            for i in range(n_plants):
                if i > 0 and random() < self.clumping_factor:
                    # it clumps to another plant
                    positions[i] = positions[randrange(i)] + np.random.normal(0, noise, 2)
                else:
                    # it gets distributed uniformly 
                    r = randrange(n_rows)
                    positions[i, 0] = row_x[r]
                    positions[i, 1] = row_y[r, 0] + random()*(row_y[r, 1] - row_y[r, 0])
                    positions[i, :] += np.random.normal(0, noise, 2)

        elif self.positioning == 'regular':
            # determine the number of clusters
            n_clumping = np.random.binomial(n_plants-1, self.clumping_factor)
            n_cluster = n_plants - n_clumping
            # determine the space between the plants
            spacing = np.random.gamma(self.regularity, 1/self.regularity, n_cluster+1)
            spacing[0]  /= 2 # the values at the borders are halved
            spacing[-1] /= 2
            # normalize the values such that the plants span all the rows
            plt_pos = np.cumsum(spacing)
            plt_pos *= n_rows / plt_pos[-1]
            plt_rowpos, plt_rowid = np.modf(plt_pos[:-1])
            plt_rowid = plt_rowid.astype(np.int)
            # place the plants with some noise
            positions[:n_cluster, 0] = row_x[plt_rowid]
            positions[:n_cluster, 1] = plt_rowpos * row_y[plt_rowid, 0] + (1-plt_rowpos) * row_y[plt_rowid, 1]
            positions[n_cluster:] = positions[np.random.randint(n_cluster, size=n_clumping), :]
            positions += np.random.normal(0, noise, (n_plants, 2))

        positions_list = []
        for i in range(n_plants):
            positions_list.append((positions[i, 1]-length/2, positions[i, 0]-width/2, 0.0))

        domain = ((-length/2,-width/2), (length/2, width/2))
        domain_area = self.length * self.width
        self.plant_density = n_plants / domain_area
        # note that the domain is in the unit given by convunit but the domain area is not
        return n_plants, domain, positions_list, domain_area
        

def agronomicStand_node(sowing_density=10, plant_density=10, inter_row=0.8,
                        noise=0, density_curve_data=None):
    return AgronomicStand(**locals())
