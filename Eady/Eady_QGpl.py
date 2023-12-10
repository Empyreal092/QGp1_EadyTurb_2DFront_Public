import sys
import numpy as np
# import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
Lx, Ly = 6*np.pi, 6*np.pi; Lz = 1
log_n = 9; Nz = 32
# log_n = 4; Nz = 4
Nx, Ny = 2**log_n, 2**log_n
delx = Lx/Nx

dealias = 3/2
stop_sim_time = 200
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
nun2 = 0.35
# nun2 = float(sys.argv[2])/100
print('nun2=%f' %nun2)
num = 1000
nu4 = delx**4

Ro = float(sys.argv[1])/1000
print('Ro=%f' %Ro)

# rand_seed = 1
rand_seed = int(sys.argv[2])

# Bases
Lzt, Lzb = 0, -Lz; Lzm = -Lz/2

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.Chebyshev(coords['z'], size=Nz, bounds=(Lzb, Lzt), dealias=dealias)

# Fields
q = dist.Field(bases=(xbasis,ybasis,zbasis) )
bt = dist.Field(bases=(xbasis,ybasis) )
bb = dist.Field(bases=(xbasis,ybasis) )
lapn2_bt = dist.Field(bases=(xbasis,ybasis) )
lapn2_bb = dist.Field(bases=(xbasis,ybasis) )
tau_lapn2_bt = dist.Field()
tau_lapn2_bb = dist.Field()

P0 = dist.Field(bases=(xbasis,ybasis,zbasis) )
P1 = dist.Field(bases=(xbasis,ybasis,zbasis) )
F1 = dist.Field(bases=(xbasis,ybasis,zbasis) )
G1 = dist.Field(bases=(xbasis,ybasis,zbasis) )

C0 = dist.Field()
tau_P0t = dist.Field(bases=(xbasis,ybasis) )
tau_P0b = dist.Field(bases=(xbasis,ybasis) )
C1 = dist.Field()
tau_P1t = dist.Field(bases=(xbasis,ybasis) )
tau_P1b = dist.Field(bases=(xbasis,ybasis) )

tau_F1t = dist.Field(bases=(xbasis,ybasis) )
tau_F1b = dist.Field(bases=(xbasis,ybasis) )
tau_G1t = dist.Field(bases=(xbasis,ybasis) )
tau_G1b = dist.Field(bases=(xbasis,ybasis) )

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])

avg =  lambda A: d3.Average(A, ('x','y'))
J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

lift_basis = zbasis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

lpH = lambda A: dx(dx(A))+dy(dy(A))
l4H = lambda A: lpH(lpH(A))
# l4H = lambda A: dx(dx(dx(dx(dx(dx(dx(dx(A))))))))+dy(dy(dy(dy(dy(dy(dy(dy(A))))))))

u = -dy(P0)+Ro*(-dy(P1)-dz(F1))
v =  dx(P0)+Ro*( dx(P1)-dz(G1))
w =         Ro*( dx(F1)+dy(G1))
b_inv =  dz(P0)+Ro*(dz(P1)+dx(G1)-dy(F1))

ut = u(z=Lzt); vt = v(z=Lzt); 
ub = u(z=Lzb); vb = v(z=Lzb); 

zeta = -dy(u)+dx(v)
div = dx(u)+dy(v)
strain = np.sqrt((dx(u)-dy(v))**2+(dx(v)+dy(u))**2)

bt_zm = bt-avg(bt)
bb_zm = bb-avg(bb)

P0t = P0(z=Lzt); P0b = P0(z=Lzb); 

KE_sub = d3.Integrate(u**2+v**2+w**2, ('x', 'y', 'z'))
PE_sub = d3.Integrate(b_inv**2, ('x', 'y', 'z'))

zdamp_b = dist.Field(bases=(xbasis,ybasis) )

loRo = 1/np.max([0.01,Ro])

# Problem
problem = d3.IVP([bt, bb, \
                  lapn2_bt, lapn2_bb, tau_lapn2_bt, tau_lapn2_bb, \
                  P0, C0, tau_P0t, tau_P0b, \
                  P1, C1, tau_P1t, tau_P1b, \
                  F1, G1, tau_F1t, tau_F1b, tau_G1t, tau_G1b
                 ], namespace=locals())

problem.add_equation("lap(P0) + lift(tau_P0t,-1) + lift(tau_P0b,-2) + C0 = 0")
problem.add_equation("dz(P0)(z=Lzt) = bt_zm"); problem.add_equation("dz(P0)(z=Lzb) = bb_zm")
# problem.add_equation("dz(P0)(z=Lzt) = bt"); problem.add_equation("dz(P0)(z=Lzb) = bb")
problem.add_equation("integ(P0) = 0")

problem.add_equation("lap(P1) + lift(tau_P1t,-1) + lift(tau_P1b,-2) - C1*loRo = grad(dz(P0))@grad(dz(P0))-2*dy(dz(P0))")
problem.add_equation("dz(P1)(z=Lzt) = avg(bt)*loRo"); problem.add_equation("dz(P1)(z=Lzb) = avg(bb)*loRo")
# problem.add_equation("dz(P1)(z=Lzt) = 0"); problem.add_equation("dz(P1)(z=Lzb) = 0")
problem.add_equation("integ(P1) = 0")

problem.add_equation("lap(F1) + lift(tau_F1t,-1) + lift(tau_F1b,-2) = 2*J(dz(P0),dx(P0))+2*dx(dx(P0))")
problem.add_equation("F1(z=Lzt) = 0")
problem.add_equation("F1(z=Lzb) = 0")
problem.add_equation("lap(G1) + lift(tau_G1t,-1) + lift(tau_G1b,-2) = 2*J(dz(P0),dy(P0))+2*dx(dy(P0))")
problem.add_equation("G1(z=Lzt) = 0")
problem.add_equation("G1(z=Lzb) = 0")

eq_damp_top = "- nun2*lapn2_bt"; eq_damp_bot = "- nun2*lapn2_bb"
# eq_diss_top = "+ nu4*l4H(bt)"; eq_diss_bot = "+ nu4*l4H(bb)"
eq_diss_top = ""; eq_diss_bot = ""
problem.add_equation("dt(bt)" + eq_damp_top + eq_diss_top + "+ Lzt*dx(bt)-vt = - ( ut*dx(bt)+vt*dy(bt) )")
problem.add_equation("dt(bb)" + eq_damp_bot + eq_diss_bot + "+ Lzb*dx(bb)-vb = - ( ub*dx(bb)+vb*dy(bb) )")

problem.add_equation("lpH(lapn2_bt) + tau_lapn2_bt = bt")
problem.add_equation("lpH(lapn2_bb) + tau_lapn2_bb = bb")
problem.add_equation("integ(lapn2_bt) = -(num/nun2)*avg(bt)")
problem.add_equation("integ(lapn2_bb) = -(num/nun2)*avg(bb)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
q['g'] = 0 
bt.fill_random('c', seed=42*rand_seed, distribution='normal', scale=1e-3) # Random noise
bt.low_pass_filter(shape=(32, 32, 1)); bt.high_pass_filter(shape=(2, 2, 1))
bb.fill_random('c', seed=88*rand_seed, distribution='normal', scale=1e-3) # Random noise
bb.low_pass_filter(shape=(32, 32, 1)); bb.high_pass_filter(shape=(2, 2, 1))

# Analysis
data_name = 'EadyQGPl_sp_%.3f_%d' %(Ro, int(sys.argv[2]))
data_name = data_name.replace(".", "d" )
snapshots = solver.evaluator.add_file_handler(data_name, sim_dt=1, max_writes=10)
snapshots.add_task(-(-bt), name='b_top')
snapshots.add_task(-(-bb), name='b_bot')
snapshots.add_task(b_inv(z=-Lz/4), name='b_mt')
snapshots.add_task(b_inv(z=-Lz/2), name='b_mm')

snapshots.add_task(zeta(z=Lzt), name='zeta_top')
# snapshots.add_task(zeta(z=-Lz/4), name='zeta_mt')
# snapshots.add_task(zeta(z=-Lz/2), name='zeta_mm')
snapshots.add_task(zeta(z=-Lz), name='zeta_bot')

snapshots.add_task(div(z=Lzt), name='div_top')
# snapshots.add_task(div(z=-Lz/4), name='div_mt')
# snapshots.add_task(div(z=-Lz/2), name='div_mm')
snapshots.add_task(div(z=-Lz), name='div_bot')

snapshots.add_task(strain(z=Lzt), name='strain_top')
# snapshots.add_task(strain(z=-Lz/4), name='strain_mt')
# snapshots.add_task(strain(z=-Lz/2), name='strain_mm')
snapshots.add_task(strain(z=-Lz), name='strain_bot')

snapshots.add_task(P0(y=Ly/2), name='P0_yslc'); snapshots.add_task(P0(z=Lzt), name='P0_zslc')
snapshots.add_task(P1(y=Ly/2), name='P1_yslc'); snapshots.add_task(P1(z=Lzt), name='P1_zslc')
# snapshots.add_task(F1(y=Ly/2), name='F1_yslc'); snapshots.add_task(F1(z=Lzt), name='F1_zslc')
# snapshots.add_task(G1(y=Ly/2), name='G1_yslc'); snapshots.add_task(G1(z=Lzt), name='G1_zslc')
# snapshots.add_task(tau_P1, name='q_1')

data_name = 'EadyQGPl_dg_%.3f_%d' %(Ro, int(sys.argv[2]))
data_name = data_name.replace(".", "d" )
diagsave = solver.evaluator.add_file_handler(data_name, sim_dt=0.1, max_writes=3e5)
diagsave.add_task(-(-KE_sub), name='KE')
diagsave.add_task(-(-PE_sub), name='PE')
diagsave.add_task(avg(bt), name='avgbt')
diagsave.add_task(avg(bb), name='avgbb')
diagsave.add_task(avg(vt), name='avgvt')
diagsave.add_task(avg(div(z=Lzt)*bt), name='avgdivbt')
diagsave.add_task(-(-C1), name='C1_sv')

# Flow properties
dt_change_freq = 10
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property(abs(ut), name='absut')
flow_cfl.add_property(abs(vt), name='absvt')
flow_cfl.add_property(abs(ub), name='absub')
flow_cfl.add_property(abs(vb), name='absvb')

print_freq = 50
flow = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow.add_property(u**2+v**2+w**2, name='KE')
flow.add_property(-(-bt), name='bt_flow')
flow.add_property(-(-vt), name='vt_flow')
# flow.add_property(bt*div_t, name='bdivt_corr')

# Main loop
timestep = 1e-7
delx = Lx/Nx

kx = xbasis.wavenumbers[dist.local_modes(xbasis)]; ky = ybasis.wavenumbers[dist.local_modes(ybasis)]; KK = np.sqrt(kx**2+ky**2)
lap4_f = lambda dt: np.exp(-nu4*dt*KK**4)
lap4_fac = lap4_f(3e-3)

try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        bt['c'] *= lap4_fac; 
        bb['c'] *= lap4_fac
        if (solver.iteration-1) % dt_change_freq == 0:
            maxU = max( 1e-10, max(flow_cfl.max('absut'),flow_cfl.max('absvt'),flow_cfl.max('absub'),flow_cfl.max('absvb')) ); 
            timestep_CFL = delx/maxU*0.5; 
            timestep = min(max(1e-7, timestep_CFL), 0.1)
        if (solver.iteration-1) % print_freq == 0:
            logger.info('Iteration=%i, Time=%f, dt=%.3e, KE=%.3f, v_t=%.3e' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE'), flow.grid_average('vt_flow')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
