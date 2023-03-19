#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>


/* .C calls */
void logcon_slope(int *sl, int *pn, double *x, double *w, double *wslr, double *p0, 
                  int *is_knot, double *phi_cur, double *phi_cur_slr, double *Fhat,
                  double *Fhatfin, double *L);

static const R_CMethodDef CEntries[] = {
    {"logcon_slope", (DL_FUNC) &logcon_slope, 12},
    {NULL, NULL, 0}
};

void R_init_logconcens(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
