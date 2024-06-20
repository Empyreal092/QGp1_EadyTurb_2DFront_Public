# Balanced simulations of Eady turbulence and strain induced frontogenesis
This is the repository for code used in the paper
- Dù, R.S., Smith, K.S. , Bühler, O., 2024. Next-order balanced model captures submesoscale physics and statistics

It contain simulation and analysis of simulations of Eady turbulence under the QG<sup>+1</sup> model and of strain induced frontogenesis under the QG<sup>+1</sup> and semigeostrophic model. 

## Instructions for Eady turbulence under QG<sup>+1</sup>
- The file [`Eady_QGpl.py`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Eady/Eady_QGpl.py) simulates Eady turbulence under the QG<sup>+1</sup> model according to the parameters in the paper.
- Use the notebook [`plot_snap_both.ipynb`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Eady/plot/plot_snap_both.ipynb) in the [`plot`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Eady/plot) folder to generate Fig. 1, Fig. 2, Fig. 3, first figure in Fig. 5, and Fig. 6 in the paper.
- Use the notebook [`plot_btend.ipynb`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Eady/plot/plot_btend.ipynb) to generate Fig. 4 in the paper.
- Use the notebook [`plot_jointpdf.ipynb`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Eady/plot/plot_jointpdf.ipynb) to generate the latter two figures in Fig. 5 in the paper.

## Instructions for strain induced frontogenesis simulations
- Use the notebook [`IC_gen.ipynb`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Front2D/IC_gen.ipynb) to generate the initial conditions for the simulations. It generate the first figure in Fig. 7 in the paper.
- The file [`2DFrontQGpl.py`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Front2D/2DFrontQGpl.py) and [`2DFrontSemiG.py`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Front2D/2DFrontSemiG.py) simulate strain induced frontogenesis under the QG<sup>+1</sup> and semigeostrophic model.
- Use the notebook [`Front_plotboth.ipynb`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Front2D/plot/Front_plotboth.ipynb) in the [`plot`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Front2D/plot) folder to generate the latter two figures in Fig. 7 in the paper.
- Use the notebook [`maxby_growth.ipynb`](https://github.com/Empyreal092/QGp1_EadyTurb_2DFront_Public/blob/main/Front2D/plot/maxby_growth.ipynb) to generate Fig. 8 in the paper.
