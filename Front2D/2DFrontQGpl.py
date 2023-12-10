# to run type "python3 2DFrontQGpl.py"
# single core code, no MPI is used

import sys
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Ly, Lz = 16, 1
Ny, Nz = 512, 16

dealias = 3/2
stop_sim_time = 6
timestepper = d3.RK111
dtype = np.float64
nu = 0

Ro = 0.3

# Bases
coords = d3.CartesianCoordinates('y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
zbasis = d3.Chebyshev(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Substitutions
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])

y, z = dist.local_grids(ybasis, zbasis)

lift_basisy = ybasis.derivative_basis(2)
lifty = lambda A, n: d3.Lift(A, lift_basisy, n)
lift_basisz = zbasis.derivative_basis(2)
liftz = lambda A, n: d3.Lift(A, lift_basisz, n)

# Fields
q = dist.Field(name='q', bases=(ybasis,zbasis))
tau_q1 = dist.Field(name='tau_q1', bases=zbasis)
tau_q2 = dist.Field(name='tau_q2', bases=zbasis)

P0 = dist.Field(name='P0', bases=(ybasis,zbasis))
tau_y1 = dist.Field(name='tau_y1', bases=zbasis)
tau_y2 = dist.Field(name='tau_y2', bases=zbasis)
tau_z1 = dist.Field(name='tau_z1', bases=ybasis)
tau_z2 = dist.Field(name='tau_z2', bases=ybasis)
tau_P0 = dist.Field(name='tau_P0')

G = dist.Field(name='G', bases=(ybasis,zbasis))
tau_Gy1 = dist.Field(name='tau_Gy1', bases=zbasis)
tau_Gy2 = dist.Field(name='tau_Gy2', bases=zbasis)
tau_Gz1 = dist.Field(name='tau_Gz1', bases=ybasis)
tau_Gz2 = dist.Field(name='tau_Gz2', bases=ybasis)

bt = dist.Field(name='bt', bases=ybasis)
bb = dist.Field(name='bb', bases=ybasis)
tau_bt1 = dist.Field(name='tau_bt1')
tau_bt2 = dist.Field(name='tau_bt2')
tau_bb1 = dist.Field(name='tau_bb1')
tau_bb2 = dist.Field(name='tau_bb2')

# Substitution
Q2 = lambda A: dy(dz(A))
dyy = lambda A: dy(dy(A))

v = -Ro*dz(G)
w =  Ro*dy(G)
vt = v(z=Lz/2); vb = v(z=-Lz/2)

Vstr = dist.Field(name='Vstr', bases=ybasis)
Vstr['g'] = -y

absdybt = abs(dy(bt))

# Problem
problem = d3.IVP([q, tau_q1, tau_q2, \
                  P0, tau_z1, tau_z2, tau_y1, tau_y2, tau_P0, \
                  G, tau_Gz1, tau_Gz2, tau_Gy1, tau_Gy2, \
                  bt, tau_bt1, tau_bt2, \
                  bb, tau_bb1, tau_bb2  \
                 ], namespace=locals())

problem.add_equation("lap(P0) + liftz(tau_z1,-1) + liftz(tau_z2,-2)  + lifty(tau_y1,-1) + lifty(tau_y2,-2) + tau_P0 = q")
problem.add_equation("integ(P0) = 0")
problem.add_equation("dz(P0)(z=-Lz/2) = bb")
problem.add_equation("dz(P0)(z=Lz/2) = bt")
problem.add_equation("dy(P0)(y=-Ly/2) = 0")
problem.add_equation("dy(P0)(y=Ly/2) = 0")

problem.add_equation("lap(G) + liftz(tau_Gz1,-1) + liftz(tau_Gz2,-2)  + lifty(tau_Gy1,-1) + lifty(tau_Gy2,-2) -2*Q2(P0) = 0")
problem.add_equation("G(z=-Lz/2) = 0")
problem.add_equation("G(z=Lz/2) = 0")
problem.add_equation("G(y=-Ly/2) = 0")
problem.add_equation("G(y=Ly/2) = 0")

problem.add_equation("dt(q) + lifty(tau_q1,-1)+lifty(tau_q2,-2) + Vstr*dy(q) - nu*dyy(q) = - ( v*dy(q)+w*dz(q) )") #!!!!
problem.add_equation("q(y=-Ly/2)=0"); 
problem.add_equation("q(y= Ly/2)=0")

problem.add_equation("dt(bt) + lifty(tau_bt1,-1)+lifty(tau_bt2,-2) + Vstr*dy(bt) - nu*dyy(bt) = -vt*dy(bt)")
problem.add_equation("dt(bb) + lifty(tau_bb1,-1)+lifty(tau_bb2,-2) + Vstr*dy(bb) - nu*dyy(bb) = -vb*dy(bb)")
problem.add_equation("bb(y=-Ly/2)=1"); problem.add_equation("bb(y=Ly/2)=-1")
problem.add_equation("bt(y=-Ly/2)=1"); problem.add_equation("bt(y=Ly/2)=-1")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
write = solver.load_state("front_IC_Ly_%d/front_IC_Ly_%d"%(Ly,Ly)+"_s1.h5", index=2, allow_missing=True)

# Analysis
spname = 'QGpq1_Ro%.2f_Ly%d_Ny%d_Nz%d_snap' %(Ro, Ly, Ny, Nz)
spname = spname.replace(".", "d" ); 
snapdata = solver.evaluator.add_file_handler(spname, sim_dt=0.01, max_writes=500)
snapdata.add_task(-(-q), name='q')
snapdata.add_task(-(-bt), name='bt')
snapdata.add_task(-(-bb), name='bb')
snapdata.add_task(-(-G), name='G')
snapdata.add_task(-(-v), name='v')
snapdata.add_task(-(-w), name='w')
snapdata.add_task(-(-absdybt), name='absdybt')

# Flow properties
dt_change_freq = 1
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property(abs(v), name='absv')
flow_cfl.add_property(abs(w), name='absw')

print_freq = 1
flow = d3.GlobalFlowProperty(solver, cadence=print_freq)
flow.add_property(abs(dy(bt)), name='absdybt')

# Timestepping
dely = Ly/Ny; delz = Lz/Nz
tp_rat = 0.1
timestep = dely*tp_rat

try:
    logger.info('Starting main loop')
    solver.step(timestep)
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % dt_change_freq == 0:
            maxV = max(1e-10,flow_cfl.max('absv'),1); maxW = max(1e-10,flow_cfl.max('absw'))
            timestep_CFL = min(dely/maxV,delz/maxW)*tp_rat
            timestep = min(max(1e-5, timestep_CFL), 1)

            maxdybt = flow.max('absdybt')
            if np.isnan(maxdybt) or maxdybt>15:
                print("b break triggered")
                break
        if (solver.iteration-1) % 10 == 0:
            maxdybt = flow.max('absdybt')
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, max|dy(bt)|=%.3e' %(solver.iteration, solver.sim_time, timestep, maxdybt) )

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()