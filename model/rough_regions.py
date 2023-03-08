"""
Create roughness input based on per-polygon values
"""
import numpy as np

def settings_to_roughness_xyz(model,settings,default_n=0.02):
    """
    (model, dict) => N x {x,y,n} array
    """
    xy=model.grid.nodes['x']
    z=np.zeros(len(xy),np.float64)
    z[:]=np.nan

    for region in settings:
        for match in model.match_gazetteer(name=region,geom_type='Polygon'):
            sel=model.grid.select_nodes_intersecting(geom=match['geom'])
            z[sel]=settings[region]
    missing=np.sum(np.isnan(z))
    if missing:
        print("%d nodes will get default of %g"%(missing,default_n))
        z[np.isnan(z)]=default_n
    return np.c_[xy,z]
    
