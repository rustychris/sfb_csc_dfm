# scripting the process of taking a DEM,
# extracting "true" per-cell averages,
# and map those to nodes.
import numpy as np
from stompy import utils
from scipy import sparse

def dem_to_cell_bathy(dem,g,fill_iters=20):
    cell_means=np.zeros(g.Ncells(),np.float64)
    for c in utils.progress(range(g.Ncells()),msg="dem_to_cell_bathy: %s"):
        #msk=dem.polygon_mask(g.cell_polygon(c))
        #cell_means[c]=np.nanmean(dem.F[msk])
        cell_means[c]=np.nanmean(dem.polygon_mask(g.cell_polygon(c),return_values=True))
    
    for _ in range(fill_iters):
        missing=np.nonzero(np.isnan(cell_means))[0]
        if len(missing)==0:
            break
        new_depths=[]
        print("filling %d missing cell depths"%len(missing))
        for c in missing:
            new_depths.append( np.nanmean(cell_means[g.cell_to_cells(c)]) )
        cell_means[missing]=new_depths
    else:
        print("Filling still left %d nan cell elevations"%len(missing))
    return cell_means
    
def dem_to_cell_node_bathy(dem,g):
    cell_z=dem_to_cell_bathy(dem,g)
    
    node_z_to_cell_z=sparse.dok_matrix( (g.Ncells(),g.Nnodes()), np.float64 )
    for c in utils.progress(range(g.Ncells())):
        nodes=g.cell_to_nodes(c)
        node_z_to_cell_z[c,nodes]=1./len(nodes)
    # A x = b
    # A: node_z_to_cell_z
    #  x: node_z
    #    b: cell_z
    # to better allow regularization, change this to a node elevation update.
    # A ( node_z0 + node_delta ) = cell_z
    # A*node_delta = cell_z - A*node_z0 
    
    node_z0=dem(g.nodes['x'])
    bad_nodes=np.isnan(node_z0)
    node_z0[bad_nodes]=0.0 # could come up with something better..
    if np.any(bad_nodes):
        print("%d bad node elevations"%bad_nodes.sum())
    b=cell_z - node_z_to_cell_z.dot(node_z0)

    # damp tries to keep the adjustments to O(2m)
    res=sparse.linalg.lsqr(node_z_to_cell_z.tocsr(),b,damp=0.05)
    node_delta, istop, itn, r1norm  = res[:4]
    print("Adjustments to node elevations are %.2f to %.2f"%(node_delta.min(),
                                                             node_delta.max()))
    final=node_z0+node_delta
    if np.any(np.isnan(final)):
        print("Bad news")
        import pdb
        pdb.set_trace()
    return final
    
    
