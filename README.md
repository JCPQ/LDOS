# LDOS
Photonic LDOS inside stratefied media. This is a set of modules that can be used to calculate the photonic density of states in layer media. The code is fully in Python, it is an implementation of the method described in the book Nano-Optics by Bert Hecht and Lukas Novotny. 

Notes:
- The dipole can never be placed in a medium with a complex dielectric constant
- The first and last layer are infinately thick and also have a real dielectric constant
- for dipoles positioned closer than 1 nm to a surface this code will not give accurate numbers. Or at least the convergence of the integral perfromed with quadgk.py should be carefully checked.
- This code is not optimized for speed but for most uses it will be adaquate
