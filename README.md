# boussinesqmeasure

This Python module approximates the solution of an optimal control problem
for the two-dimensional steady Boussinesq equation with regular Borel
measures as controls. The problem being considered is the following:

    min (1/2)|u - ud|^2 + (1/2)|z - zd|^2 + a|q| + b|y|
    subject to the state equation:
        - nu Delta u + (u.Grad) u + Grad p = zg + q     in Omega
                                     div u = 0          in Omega
              - kappa Delta z + (u.Grad) z = y          in Omega
                                         u = 0          on Gamma
                                         z = 0          on Gamma
    over all controls q in M(Omega) x M(Omega) and y in M(Omega).

If you find these codes useful, you can cite the manuscript as:
> Peralta, G., Optimal Borel measure controls for the two-dimensional
  stationary Boussinesq system, to appear in ESAIM: Control, Optimisation
  and Calculus of Variations, 2022.
