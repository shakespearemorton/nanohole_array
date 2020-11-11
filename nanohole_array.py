import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep.materials import Au

def simulation(square,THICKNESS,cx,cy,cz):
    #----------------------------Variables------------------------------
    period = square 							#um
    PADDING = 0.70 							#um
    fmin = 1/.800 							#maximum wavelength
    fmax = 1/.300 							#minimum wavelength
    fcen = (fmax+fmin)/2 						#set centre of gaussian source
    df = fmax-fmin 							#set width of gaussian source
    nfreq = 50 								#number of frequencies between min and max
    dpml = 0.25 							#thickness of PML (top and bottom) um
    resolution = 100 							#pixels/um
    BASE = PADDING-THICKNESS 						#metal thin film is set on a PDMS base
    sz = THICKNESS + 2 * PADDING + 2 * dpml 				#size of simulation
    sx = period 							#size of simulation
    sy = period 							#size of simulation
    box = 0.010 							#optimised size of radiation monitoring box set around dipole
    cell = mp.Vector3(sx, sy, sz)
    
    #define geometry
    slab = mp.Block(size=mp.Vector3(1e20, 1e20, THICKNESS), center=mp.Vector3(0, 0, -THICKNESS/2), material=Au) 			#Gold thin film
    hole = mp.Block(size=mp.Vector3(0.145,0.145,THICKNESS), center=mp.Vector3(0, 0, 0)) 						#nanohole
    slab3 = mp.Block(mp.Vector3(1e20, 1e20, BASE), center=mp.Vector3(0, 0, -0.5*sz+dpml+0.5*BASE), material=mp.Medium(index=1.53))	#pdms base
    geometry = [slab, hole, slab3]
    
    #--------------------------Simulation Parameters----------------------------------------
    
    #define Gaussian plane wave Ez polarised
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(cx, cy, cz)
        )
    ]
    
    #define pml layers (Absorber is a type of PML that helps when there are Bloch Wave SPP modes. Placed in substrate)
    pml_layers = [mp.Absorber(thickness=dpml, direction=mp.Z, side=mp.Low),mp.PML(thickness=dpml, direction=mp.Z, side=mp.High)]

    #sets the simulation without the substrate so that it is a homogeneous environment
    sim = mp.Simulation(cell_size=cell,
            boundary_layers=pml_layers,
            sources=sources,
            resolution=resolution,
            k_point=mp.Vector3(0,0,0))
    
    #power monitors around the simulation
    #power monitor around the dipole
    dipole_box = sim.add_flux(fcen, df, nfreq,
                              mp.FluxRegion(center=mp.Vector3(cx+0.5*box,cy,cz), size=mp.Vector3(0,box,box),direction=mp.X,weight=+1),
                              mp.FluxRegion(center=mp.Vector3(cx-0.5*box,cy,cz), size=mp.Vector3(0,box,box),direction=mp.X,weight=-1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy,cz+0.5*box), size=mp.Vector3(box,box,0),direction=mp.Z,weight=+1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy,cz-0.5*box), size=mp.Vector3(box,box,0),direction=mp.Z,weight=-1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy+0.5*box,cz), size=mp.Vector3(box,0,box),direction=mp.Y,weight=+1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy-0.5*box,cz), size=mp.Vector3(box,0,box),direction=mp.Y,weight=-1))

    #power monitor on the surface of the thin film
    rad_box = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0,0,0), size=mp.Vector3(sx,sy,0),weight=-1))
    
    #power monitors surrounding 'free space'
    top_box = sim.add_flux(fcen, df, nfreq, 
                           mp.FluxRegion(center=mp.Vector3(0,0,0.5*sz-dpml), size=mp.Vector3(sx,sy,0),weight=+1),
                           mp.FluxRegion(center=mp.Vector3(0,0.5*sy,0.25*sz), size=mp.Vector3(sx,0,0.5*sz-dpml),weight=+1),
                           mp.FluxRegion(center=mp.Vector3(0,-0.5*sy,0.25*sz), size=mp.Vector3(sx,0,0.5*sz-dpml),weight=-1),
                           mp.FluxRegion(center=mp.Vector3(0.5*sx,0,0.25*sz), size=mp.Vector3(0,sy,0.5*sz-dpml),weight=+1),
                           mp.FluxRegion(center=mp.Vector3(-0.5*sx,0,0.25*sz), size=mp.Vector3(0,sy,0.5*sz-dpml),weight=-1))
    
    #run simulation until the source has decayed "fully"
    sim.run(until_after_sources=mp.stop_when_fields_decayed(5, mp.Ez, mp.Vector3(cx,cy,cz), 1e-8))
    
    #collect radiation information
    init_rad = np.asarray(mp.get_fluxes(rad_box))               
    init_dipole = np.asarray(mp.get_fluxes(dipole_box))
    init_top = np.asarray(mp.get_fluxes(top_box))

    
    sim.reset_meep()
    
    #run simulation again with the substrate (inhomogeneous environment)
    sim = mp.Simulation(cell_size=cell,
            boundary_layers=pml_layers,
            sources=sources,
            geometry=geometry,
            resolution=resolution,
            k_point=mp.Vector3(0,0,0))
    
    #power monitors around the dipole
    dipole_box2 = sim.add_flux(fcen, df, nfreq,
                              mp.FluxRegion(center=mp.Vector3(cx+0.5*box,cy,cz), size=mp.Vector3(0,box,box),direction=mp.X,weight=+1),
                              mp.FluxRegion(center=mp.Vector3(cx-0.5*box,cy,cz), size=mp.Vector3(0,box,box),direction=mp.X,weight=-1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy,cz+0.5*box), size=mp.Vector3(box,box,0),direction=mp.Z,weight=+1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy,cz-0.5*box), size=mp.Vector3(box,box,0),direction=mp.Z,weight=-1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy+0.5*box,cz), size=mp.Vector3(box,0,box),direction=mp.Y,weight=+1),
                              mp.FluxRegion(center=mp.Vector3(cx,cy-0.5*box,cz), size=mp.Vector3(box,0,box),direction=mp.Y,weight=-1))
    
    rad_box2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0,0,0), size=mp.Vector3(sx,sy,0),weight=-1))
    
    top_box2 = sim.add_flux(fcen, df, nfreq, 
                           mp.FluxRegion(center=mp.Vector3(0,0,0.5*sz-dpml), size=mp.Vector3(sx,sy,0),weight=+1),
                           mp.FluxRegion(center=mp.Vector3(0,0.5*sy,0.25*sz), size=mp.Vector3(sx,0,0.5*sz-dpml),weight=+1),
                           mp.FluxRegion(center=mp.Vector3(0,-0.5*sy,0.25*sz), size=mp.Vector3(sx,0,0.5*sz-dpml),weight=-1),
                           mp.FluxRegion(center=mp.Vector3(0.5*sx,0,0.25*sz), size=mp.Vector3(0,sy,0.5*sz-dpml),weight=+1),
                           mp.FluxRegion(center=mp.Vector3(-0.5*sx,0,0.25*sz), size=mp.Vector3(0,sy,0.5*sz-dpml),weight=-1))
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(5, mp.Ez, mp.Vector3(cx,cy,cz), 1e-8))
    
    rad = np.asarray(mp.get_fluxes(rad_box2))
    dipole = np.asarray(mp.get_fluxes(dipole_box2))
    top = np.asarray(mp.get_fluxes(top_box2))

    #Output data 
    data = np.zeros((6,len(rad)))
    data[0,:] = init_rad
    data[1,:] = rad
    data[2,:] = init_dipole
    data[3,:] = dipole
    data[4,:] = init_top
    data[5,:] = top
        
    np.savetxt(repr(cz)+'.txt',data)
    
    return()


THICKNESS = 0.045                               #thickness of the metal for the substrate
square = 0.500                                  #Period of the square array
cx = -0.5*0.145                                 #Position of highest enhancement for a Ez Polarised source
cy = 0
cz = np.linspace(0.005,0.100,25)
for i in cz:
  simulation(square,THICKNESS,cx,cy,cz[i])
