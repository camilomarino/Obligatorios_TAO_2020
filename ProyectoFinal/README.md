# Proyecto Final TAO (Teoría y Algoritmia de Optimización) 2020

### Informe:
El documento que explica los experimentos y resultados se puede encontrar en:
- [Letra Informe](pdfs/informe_final.pdf)

### Scripts:
- [nnls](nnls.py): Implementación del algoritmo NNLS.
- [pgd](pgd.py): Implementación del algoritmo PGD para la resolucion del problema de optimizacion planteado.
- [apgd](apgd.py): Implementación del algoritmo A-PGD para la resolucion del problema de optimizacion planteado.

### Notebooks
- [Creacion de matrices](Create_vectors.ipynb): Se crean las matrices D y X para resolver los problemas de optimización.
- [Resolución del problema sin regularización](Solve_simple.ipynb): Se resuelve mediante A-PGD.
- [Resolución del problema S2K como regularización](Solve_regularization.ipynb): Se resuelve mediante A-PGD.
- [Resolución del problema S2K como restriccion](Solve_restriction.ipynb): Se resuelve mediante A-PGD.
- [Evaluación](Evaluate_results.ipynb): Se evaluan las metricas al utilizar las distintas matrices A que se estiman con los distintos modelados.