# WaveFunctionCollapse

La compilation du solveur CUDA est automatiquement activée si un compilateur CUDA est détecté par cmake.
Pour compiler avec le solveur cuda, ajouter l'option `-DCMAKE_CUDA_ARCHITECTURES="..."` à l'invocation
de cmake, les valeurs possibles sont documentées [ici](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html).

Le programme a été compilé avec gcc/11.3 et cuda/11.7 (et cuda/12.0) sur le
supercalculateur ROMEO, ainsi que les compilateurs gcc 13.2.1 et clang 17.0.6
en local.

4 solveurs sont disponibles :
- `cpu`: solveur séquentiel sur cpu
- `omp`: solveur parallèle (une seed à la fois) sur cpu
- `omp_par`: solveur parallèle (plusieurs seeds à la fois, chaque seed en séquentiel) sur cpu
- `cuda`: solveur parallèle (une seed à la fois) sur gpu

Au lancement du programme, l'option `-p` permet de préciser le nombres de threads utilisés par les versions OpenMP
