# Put bathy on the combined grid
import six
import os
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import stompy.plot.cmap as scmap
import numpy as np
from stompy import utils
from stompy.spatial import field
import stompy.model.delft.dflow_model as dfm
import logging
logging.root.setLevel(logging.INFO)

turbo=scmap.load_gradient('turbo.cpt')

##
fn_src='sfb_csc-v00-edit013.nc'
g=unstructured_grid.UnstructuredGrid.read_ugrid(fn_src)

UnstructuredGrid=unstructured_grid.UnstructuredGrid

@utils.add_to(g)
def create_dual(self,center='centroid',create_cells=False,
                remove_disconnected=False,remove_1d=True,
                expand_to_boundary=False):
    """
    Return a new grid which is the dual of this grid. This
    is robust for triangle->'hex' grids, and provides reasonable
    results for other grids.

    remove_1d: avoid creating edges in the dual which have no
      cell.  This happens when an input cell has all of its nodes
      on the boundary.

    expand_to_boundary: in contrast to remove_1d, this adds 
      edges to these 1d features to make them 2d, more or less
      taking the implied rays and intersecting them with the
      boundary of the original grid. If specifically set to
      "exact" then boundary nodes are included as nodes in their
      own dual cells. This has the effect of matching the boundary
      of the dual and the original.
    """
    if remove_disconnected and not create_cells:
        # anecdotal, but remove_disconnected calls boundary_linestrings,
        # which in turn needs cells.
        raise Exception("Creating the dual without cells is not compatible with removing disconnected")
    gd=UnstructuredGrid()

    if center=='centroid':
        cc=self.cells_centroid()
    else:
        cc=self.cells_center()

    gd.add_node_field('dual_cell',np.zeros(0,np.int32))
    if expand_to_boundary:
        gd.add_node_field('dual_edge',np.zeros(0,np.int32))
    gd.add_edge_field('dual_edge',np.zeros(0,np.int32))

    if expand_to_boundary:
        remove_1d=False
        
    # precalculate mask of boundary nodes
    if remove_1d or expand_to_boundary:
        e2c=self.edge_to_cells()
        boundary_edge_mask=e2c.min(axis=1)<0
        boundary_nodes=np.unique(self.edges['nodes'][boundary_edge_mask])
        boundary_node_mask=np.zeros(self.Nnodes(),np.bool8)
        boundary_node_mask[boundary_nodes]=True
        # below, if a cell's nodes are all True in boundary_node_mask,
        # it will be skipped

    cell_to_dual_node={}
    for c in self.valid_cell_iter():
        # dual_cell is redundant if remove_1d is False.
        if remove_1d:
            nodes=self.cell_to_nodes(c)
            if np.all(boundary_node_mask[nodes]):
                continue
        node_idx=gd.add_node(x=cc[c],dual_cell=c)
        cell_to_dual_node[c]=node_idx

    e2c=self.edge_to_cells()

    if expand_to_boundary:
        boundary_edge_to_dual_node=-np.ones(self.Nedges(),np.int64)
        edge_center=self.edges_center()

    for j in self.valid_edge_iter():
        if e2c[j].min() < 0:
            if expand_to_boundary:
                # Boundary edges *also* get nodes at their midpoints
                boundary_edge_to_dual_node[j] = dnj = gd.add_node(x=edge_center[j],
                                                                  dual_edge=j)
                # And induce a dual edge from the neighboring cell's dual
                # node to this edge's midpoint
                dnc=cell_to_dual_node[e2c[j,:].max()]
                dj=gd.add_edge(nodes=[dnj,dnc],dual_edge=j)
            else:
                continue # boundary
        elif remove_1d and np.all(boundary_node_mask[self.edges['nodes'][j]]):
            continue # would create a 1D link
        else:
            # Regular interior edge
            dn1=cell_to_dual_node[e2c[j,0]]
            dn2=cell_to_dual_node[e2c[j,1]]

            dj_exist=gd.nodes_to_edge([dn1,dn2]) 
            if dj_exist is None:
                dj=gd.add_edge(nodes=[dn1,dn2],dual_edge=j)

    if expand_to_boundary:
        # Nodes also imply an edge in the dual -- and maybe even two
        # edges if we want this edge to go through the node

        # This will make the dual's boundary coincident with the
        # original boundary.
        # Map self.nodes on the boundary to a half-edge in the dual
        # that we'll later use to construct the corresponding cell
        expanded_node_to_dual_halfedge={}
        
        for n in np.nonzero(boundary_node_mask)[0]:
            jbdry=[j for j in self.node_to_edges(n) if boundary_edge_mask[j]]
            # jbdry could be a multiple of 2 if there are multiple boundaries
            # tangent to each other at n.
            assert len(jbdry)==2,"Not ready for coincident boundaries (at %s)"%(self.nodes['x'][n])
            dnodes=boundary_edge_to_dual_node[jbdry]
            assert np.all(dnodes>=0)

            # Figure out orientation to prep the halfedge for cell creation.
            # here we have jbdry, two boundary edges from g, incident to n.
            # Arbitrarily focus on jbdry[0]
            orient=0
            # First include orientation of jbdry[0],
            # so that orient reflects an interior facing half
            # edge on jbdry[0]
            if e2c[jbdry[0],0]<0: # left side of jbdry[0] is outside, 
                orient=1-orient
            if self.edges['nodes'][jbdry[0],0]==n:
                pass
            elif self.edges['nodes'][jbdry[0],1]==n:
                # orientation for jbdry[0] was correct,
                # but direction of dj1, being dn->dnodes[0],
                # is opposite of jbdry[0]
                orient=1-orient
            else:
                raise Exception("Node wasn't found on edge??")
            
            if expand_to_boundary=='exact':
                dn=gd.add_node(x=self.nodes['x'][n])
                # Probably need to link dn and or these edges back to original
                # node to help with reconstruction of cells below
                dj1=gd.add_edge(nodes=[dn,dnodes[0]])
                dj2=gd.add_edge(nodes=[dn,dnodes[1]])
                    
                # Trying to decide orientation of this halfedge:
                # orient=0 means the half edge is from dn to the middle of
                # jbdry[0]
                expanded_node_to_dual_halfedge[n]=gd.halfedge(dj1,orient)
            else:
                dj=gd.add_edge(nodes=dnodes)
                # This gets flipped _again_, because for 'exact', we create
                # an edge dn->dnodes[0], but here we create an edge
                # dnodes[0]->dnodes[1]
                expanded_node_to_dual_halfedge[n]=gd.halfedge(dj,1-orient)

    # The usual way to construct the cells is:
    # 1. iterate over the nodes in self
    #    skip boundary nodes
    #    in self, map each node to neighboring cells, and each of those
    #     cells to a dual node. angle-sort these dual nodes, and we're done.
    
    if create_cells:
        # to create cells in the dual -- these map to interior nodes
        # of self.
        max_degree=10 # could calculate this.
        e2c=self.edge_to_cells()
        gd.modify_max_sides(max_degree)

        gd.add_cell_field('source_node',np.zeros(gd.Ncells(),np.int32)-1,
                          on_exists='overwrite')

        for n in self.valid_node_iter():
            if n%1000==0:
                print("%d/%d dual cells"%(n,self.Nnodes())) # not exact
            js=self.node_to_edges(n)
            if np.any(e2c[js,:]<0): # boundary
                if expand_to_boundary:
                    he=expanded_node_to_dual_halfedge[n]
                    # Get cycle from he
                    dual_cycles=gd.find_cycles(starting_edges=[he])
                    if len(dual_cycles)==1:
                        gd.add_cell(nodes=dual_cycles[0],source_node=n)
                    else:
                        #assert len(dual_cycles)==1
                        print("Half edge %s was not forthcoming"%he)
                        import pdb
                        pdb.set_trace()
                        return gd
            else:
                tri_cells=self.node_to_cells(n) # i.e. same as dual nodes
                dual_nodes=np.array([cell_to_dual_node[c] for c in tri_cells])

                # but those have to be sorted.  sort the tri cell centers, same
                # as dual nodes, relative to tri node
                diffs=gd.nodes['x'][dual_nodes] - self.nodes['x'][n]
                angles=np.arctan2(diffs[:,1],diffs[:,0])
                dual_nodes=dual_nodes[np.argsort(angles)]
                gd.add_cell(nodes=dual_nodes,source_node=n)

        # flip edges to keep invariant that external cells are always
        # second.
        e2c=gd.edge_to_cells()
        to_flip=e2c[:,0]<0
        for fld in ['nodes','cells']:
            gd.edges[fld][to_flip] = gd.edges[fld][to_flip][:,::-1]

        # the original node locations are a more accurate orthogonal
        # center than what we can calculate currently.
        gd.cells['_center']=self.nodes['x'][gd.cells['source_node']]

    if remove_disconnected:
        gd.remove_disconnected_components()

    return gd

gd=g.create_dual(create_cells=True,expand_to_boundary="exact")

# Fill in bathy from DEMs

# Sources:
#   DWR TIFFs
data_root="/home/rusty/data/"

# 2021-09-22: Downloaded all of the current Delta bathy DEMs:
#   dem_bay_delta_10m_20201207.tif
#   dem_calaveras_rvr_2m_20120902.tif
#   dem_ccfb_south_delta_san_joaquin_rvr_2m_20200625.tif
#   dem_columbia_cut_2m_20120911.tif
#   dem_delta_10m_20201207.tif
#   dem_false_rvr_piper_sl_fisherman_cut_2m_20171109.tif
#   dem_montezuma_sl_2m_20200909.tif
#   dem_north_delta_2m_20201130.tif
#   dem_sac_rvr_decker_isl_2m_20150914.tif
#   dem_turner_cut_2m_20120907.tif
#   dem_yolo_2m_20200505.tif

# Other sources:
#   CSC merge_2m-20190122.tif (for Ulatis)
mrf=field.MultiRasterField( [ os.path.join(data_root,'bathy_dwr/gtiff/*.tif'),
                              ('/home/rusty/src/csc/bathy/merged_2m-20190122.tif',-10) ],
                            min_valid=-1e5,
                            max_count=10)

# Follow what CSC used
# Maybe 5 minutes, and a lot of RAM
z_dualcell=dfm.dem_to_cell_bathy(mrf,gd)

#g.add_node_field('node_z_bed
z_node=z_dualcell[ np.argsort(gd.cells['source_node']) ]

##

# Two nodes that fall in a quarry that I'd rather not copy over:
omit_xy=[(548306.0369393384, 4204530.249395246),
         (548129.5676682167, 4204634.737779463)]
to_copy=np.ones(g.Nnodes(),np.bool8)
for xy in omit_xy:
    to_copy[g.select_nodes_nearest(xy)]=False

g.nodes['depth'][to_copy]=z_node[to_copy]

# Rename to make it clearer that it's elevation
g.add_node_field( 'node_z_bed', g.nodes['depth'], on_exists='overwrite')

##

plt.figure(1).clf()
g.contourf_node_values(g.nodes['depth'],np.linspace(-15,5,30),cmap=turbo,extend='both')
g.plot_edges(color='k',alpha=0.3,lw=0.4)

plt.axis('equal')
plt.axis('off')
plt.gcf().tight_layout()

##

# Make a DFM-ish output:
fn_dest=fn_src.replace('.nc','') + '_net.nc'
g.write_dfm(fn_dest, node_elevation='node_z_bed', overwrite=True)

fn_dest=fn_src.replace('.nc','') + '_bathy.nc'
g.write_ugrid(fn_dest, overwrite=True)

