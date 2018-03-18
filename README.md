# LDOS
This is a set of modules that can be used to calculate the photonic local density of states (LDOS) in layered media. The LODS can be used to calculate the fluorescence lifetime of molecules or atoms that are placed on top or in layerd media. The code is fully in Python. The file test_LDOS_MultiLayer.py is an example script that shows how to use the modules. Some of the modules contain functions not directly used in calculating the LDOS but that could be used for other calculations such as the collection eficiency of light from fluorophores or the LDOS in more simple geometries (not having many layers).

Sources that were particularily useful to make this code
[1] Principles of Nano Optics by L.Novotny and B.Hecht, 2nd edition
[2] Theory of the radiation of dipoles within a multilayer system, Polerecjy, Harmle and Maccraith Applied Optics Vol 39 page 3968 (2000)
[3] Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings, Lifeng Li, J. opt. Soc. Am A 13 1024 (1996)
[4] Green’s tensor technique for scattering in two-dimensional stratified media Michael Paulus and Olivier J. F. Martin Phys. Rev. E 63, 066615 – Published 29 May 2001

Notes for use:
- The first and last layer are infinately thick and shopuld have a real dielectric constant.
- For dipoles positioned closer than 1 nm to a surface this code will not give accurate numbers. Or at least the convergence of the integral perfromed with quadgk.py should be carefully checked.
- This code is not optimized for speed but for most uses it will be adaquate.
- The dipole should never be placed in a medium with a complex dielectric constant. Physically this leads to quenching but the integrals in this code will still give an answer which will be wrong. If the fluorophore is in an absorbing medium you need to read up on "local field effects" and this code will not help you. 

