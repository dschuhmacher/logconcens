/*

   logcon_slope.c

   Revision: 0.17.1   Date: 2022/10/17   
   (maintenance fix: replace old-style K&R functions)

   Code by Dominic Schuhmacher
   Ported and extended based on Matlab code by Lutz Duembgen and R code by Kaspar Rufibach

*/


#include <math.h>
#include <R.h>
#include <R_ext/Lapack.h>
#include <R_ext/Utils.h>
/* the last include makes the function   void R_CheckUserInterrupt(void)  available, which
   checks for user interrupt and allows compiled code to be stopable at certain times */


/* Note: the State structure includes many "temporary variables", because I suspect
   that the many calls to Calloc inside the loops would slow down the programm */
typedef struct State {
  int sl;   /* 1 if a rhs slope is included, 0 otherwise (slope weight ignored); to be extended to two slopes */
  int n, rn;   /* number of points and number of knots (=reduced number of points), respectively
                  number of knots refers to the _new quantities if and is always equal to 
                  the number of 1s in is_knot; new and old objects are the same at the beginning
                  of each pass through the outer while loop in logcon, so then it doesn't matter */        
  double L;    /* likelihood for current(?) phi */
  double p0;
  /* vectors of length n */
  int *is_knot;   /* binary (unsigned char would use less memory, but I don't think it really matters */
  int *is_knot_old;
  double *x;
  double *dx;
  double *w;
  double *w_slr;         /* length 1 */
  double *phi_cur;
  double *phi_cur_slr;   /* length 1 */
  double *phi_new;
  double *phi_new_slr;   /* length 1 */
  double *conv_cur;      /* in the rslope version conv_cur[n-1] has meaning, not just = 0 */
  double *conv_new;      /* same for conv_new[n-1] */
  double *H;
  double *xtmp;   /* this is a helper vector used to compute H, only used in local mle;
                     carried through everything to save computation time */
} State;

/* RState is "only" used in localmle; it deals with the reduced (aka local) quantities.
   It does not seem to be a good idea to create a single RState variable throughout, because
   the length of the vectors may change with each call to localmle; it is never larger than
   n but typically much much much smaller */
typedef struct RState {
  int sl;   /* 1 if a rhs slope is included, 0 otherwise (slope weight ignored); to be extended to two slopes */
  int rn; 
  double L;    /* likelihood for current(?) phi */
  double p0; 
  int *knotlist;   /* the indices of the knots, i.e. the indices i where state.is_knot[i] == 1 */
  double *rx, *rdx;
  double *rw;
  double *rw_slr;         /* length 1 */
  double *rphi_cur;
  double *rphi_cur_slr;   /* length 1 */
  double *rphi_new;
  /* double *rphi_new_slr;   length 1, probably not needed; CHECK */
  double *rgrad;
  double *rmhess_diag;
  double *rmhess_sub;  /* the main and the sub/superdiagonal of "rmhess" respectively 
                          note that all other entries of mhess are zero 
                          note that these two vector hold most of the time some auxiliary
                          output from the LAPACK routine which is not important for us */
  double *temp;   /* two more vectors that are needed for the computation of mhess */
  double *b;
} RState;



#define PRECMAIN (1.0e-10)
#define PRECNEWTON (1.0e-7)
/*  #define MAX(A,B) ((A)>(B) ? (A) : (B))
#define MIN(A,B) ((A)<(B) ? (A) : (B))  */



void localmle_slope(State *state);
void mle_slope(RState *rs);

double J00(double r, double s);
double J10(double r, double s);
double J11(double r, double s);
double J20(double r, double s);
double Local_LL_slope(int rn, double *rx, double *rdx, double *rw, double *rw_slr, double *rphi);
         /* LOCAL refers to "fixed knots"; REDUCED refers to objects 
            considered only at these knots */
double Local_LL_rest_slope(RState *rs);
void LocalNormalize_slope(RState *rs);
void LocalReduce_slope(State *state, RState *rs);
void LocalExtend_slope(RState *rs, State *state);
void LocalConvexity_slope(RState *rs, State *state);

/* void diff(int n, double *x, double *dx); */
double amax(int n, double *a);
double amaxplus(int n, double *a, int *loc);
double amaxabs(int n, double *a);


/* ------------ The main function ----------------------------- */

/* function name as called in .C */
/* N.B.: logcon implicitly returns phi_cur; localmle implicitly returns state.phi_new,
   mle implicitly return rs.phi_cur; rather confusing (cur vs. new I mean), but there
   seems to be no easy way around it (the confusing thing is really that mle returns
   rs.phi_cur instead of rs.phi_new, which is based on the idea that if the dirderiv is
   not significantly bigger than 0 we don't bother to return the new phi;
   it wouldn't hurt if we did, would it?  */
void logcon_slope(int *sl, int *pn, double *x, double *w, double *wslr, double *p0, 
                  int *is_knot, double *phi_cur, double *phi_cur_slr, double *Fhat,
                  double *Fhatfin, double *L)
  /* careful: w is assumed to be already normalised so that it sums to 1 */
  /* inputs */
  //int *sl;
  // int *pn;
  // double *x, *w, *wslr; 
  // double *p0; 
  /* outputs */
  // int *is_knot; 
  // double *phi_cur, *phi_cur_slr;
  // double *Fhat;
  // double *Fhatfin;
  // double *L;
{ 
  int i, j, iter1, n;
  int argmax;  /* used for H (as argmax) */
  int lambdacounter;
  double maxH, sabsH = 0;
  double lambda, lambdatmp;   /* lambda max of the conv quotients, lambdatmp is used in the computation */
  /*  double maxconv, maxabsconv = 0; */
  int *lambdalist;
  State state;  
  /* double dderiv, deriv1, deriv2, temp; trying, delete if not needed */

  /* inputs/outputs */
  state.sl = *sl;
  state.p0 = *p0;
  state.n = n = *pn;
  state.x = x; 
  state.w = w;
  state.w_slr = wslr;
  state.is_knot = is_knot;  
  state.phi_cur = phi_cur; 
  state.phi_cur_slr = phi_cur_slr; /* initialize the state *pointers*;
			          so output is written in the right place; is_knot and phi_cur
			          without state are not needed anymore */
  /* Note that for Fhat and L we do not bother to have corresponding state "versions" */

  /* scratch space */
  lambdalist =  (int *) R_alloc((long) n, sizeof(int));
  state.is_knot_old = (int *) R_alloc((long) n, sizeof(int));
  state.dx = (double *) R_alloc((long) n, sizeof(double));
  state.phi_new = (double *) R_alloc((long) n, sizeof(double));
  state.phi_new_slr = (double *) R_alloc((long) 1, sizeof(double));
  state.conv_cur = (double *) R_alloc((long) n, sizeof(double));
  state.conv_new = (double *) R_alloc((long) n, sizeof(double));
  state.H = (double *) R_alloc((long) n, sizeof(double));
  state.xtmp = (double *) R_alloc((long) n, sizeof(double));

  for (i = 0; i < n-1; i++) {
    state.dx[i] = state.x[i+1] - state.x[i];
  }

  /* initialize some more state variables */
  /* starting phi in the slope case: just one knot in tau_o, optimal phi for this situation
     could be computed explicitly, but for the situation without slope we have to compute
     numerically anyway, so we do this here to and check afterwards */
 
  /* start with default 1-knot phi */
  if (state.sl == 1) {
    state.rn = 1;
    state.is_knot[0] = 1;
    state.phi_cur[0] = log(1.0 - state.p0);
    state.phi_cur_slr[0] = -1;
    /* this is a trick: we would like to use the optimal slope from DHR, p.22 for consistency reasons
       (consistency with latter passes of mle_slope, that is), but it's to annoying to compute before
       making the first reduction step. Therefore we set it to an arbitrary value. This value
       will never be actually used and at the end of the first pass through mle_slope it is set to
       a correct value. So just as a warning until this time state.phi_cur_slr[0] is just arbitrary. */
    for (i = 1; i < n; i++) {
      state.is_knot[i] = 0;
      state.phi_cur[i] = state.phi_cur[0] + state.x[0] - state.x[i];
         /* instead of call to LocalNormalize */
    }
  }
  else {
    state.w_slr[0] = 0;
    /* all hell breaks loose if state.sl == 0, but state.w_slr > 0
       so we better catch this on a low level */
    state.rn = 2;   
    state.is_knot[0] = 1;
    state.is_knot[n-1] = 1;
    state.phi_cur[0] = log(1.0 - state.p0) - log(state.x[n-1] - state.x[0]);
    state.phi_cur[n-1] = log(1.0 - state.p0) - log(state.x[n-1] - state.x[0]);
    state.phi_cur_slr[0] = R_NegInf;   
                         /* I use array notation although length is only 1,
                            because I'm prone to forgetting the * in pointer notation */
                         /* Recommended be R-exts document to represent IEEE negative infinity */ 
    for (i = 1; i < n-1; i++) {
      state.is_knot[i] = 0;
      state.phi_cur[i] = log(1.0 - state.p0) - log(state.x[n-1] - state.x[0]); 
                         /* instead of call to LocalNormalize */
    }
  }

  /* starting phi_cur with right slope based on the idea that roughly a mass of 1/(1+w_slr) should
     be between x[0] and x[n-1] and the remaining mass w_slr/(1+w_slr) should be to the right
     of x[n-1] (where we have the slope parameter). It is not clear if this is the most sensible
     starting value   */  

  /* note: the following *is* needed: we compute phi_new (does in general *not* stay the same) H and L
     (which is not needed for further computations, but if we alread arrive at the final solution here
     we need the L for returning;
     only computation of conv_new could be skipped (it is zero everywhere)  */
  localmle_slope(&state);  /* should return (implicitly) updated phi_new, phi_new_slr as well as
			updated L, conv_new and H */

  for (i = 0; i < n; i++) {
    state.phi_cur[i] = state.phi_new[i];
    state.conv_cur[i] = state.conv_new[i];
  }
  state.phi_cur_slr[0] = state.phi_new_slr[0];
  iter1 = 1;
  for (i = 0; i < n; i++) {
    sabsH += fabs(state.H[i]);
  }
  maxH = amaxplus(n, state.H, &argmax);
  
  while ((iter1 < 500) && (maxH > PRECMAIN * (sabsH/n))) {
    R_CheckUserInterrupt();
    iter1++;
    for (i = 0; i < n; i++) {
      state.is_knot_old[i] = state.is_knot[i];
    }
    state.is_knot[argmax] = 1;
    state.rn++;
    /* it is not possible that state.is_knot[argmax] was already 1 because
       state.H[argmax] is strictly positive, and we have set state.H at knot-indices to exactly zero  */
 
    /* local optimisation after removal of 1 active constraint */
    localmle_slope(&state);

    while (amax(n,state.conv_new) > PRECMAIN * amaxabs(n,state.conv_new)) {
      /* note that conv_new has values 0 at left and right end and in between at index i it
         is the slope difference around x[i] (hence zero unless x[i] is a knot) */
         
      /* search for the smallest index i with positive conv_new[i]; I don't think that much is
         gained by other programming strategies */
      for (i = 0; i < n; i++) {
        if (state.conv_new[i] > 0) {
          break;
        }
      }
      lambda = state.conv_cur[i]/(state.conv_cur[i] - state.conv_new[i]);
      lambdalist[0] = i;
      lambdacounter = 1;
      for (j = i+1; j < n; j++) {
        if (state.conv_new[j] > 0) {
          lambdatmp = state.conv_cur[j]/(state.conv_cur[j] - state.conv_new[j]);
          if (lambdatmp < lambda) {
            lambdalist[0] = j;
            lambdacounter = 1;
            lambda = lambdatmp;
          } 
          else if (lambdatmp == lambda) {
            /* very rare; in theory impossible(?) in practice due to finite precision
               it has happened in one pass of the loop in one out of tens of thousands simulated datasets */
            lambdalist[lambdacounter] = j; 
            lambdacounter++;           
          }
        }
      }
      for (i = 0; i < lambdacounter; i++) {
        state.is_knot[lambdalist[i]] = 0;
        state.rn--;
        /* it is not possible that state.is_knot[lambdalist[i]] was already 0 because
           state.conv_new[lambdalist[i]] > 0, and we have set state.conv_new at at non-knot indices to 
           exactly 0 */
        /* NEW IN RSLOPE VERSION: this may remove the knot at index n-1 */
      }
      /* if (lambdacounter > 1) { Rprintf("Note: %d points removed. \n", lambdacounter); }  */
      /* would check for the very rare event described above; but doesn't cause problems
         so the user doesn't have to worry */ 

      /* the following sets a new phi_cur and replaces the call to LocalConvexity in RD-program */
      for (i = 0; i < n; i++) {
        state.phi_cur[i] = (1-lambda) * state.phi_cur[i] + lambda * state.phi_new[i]; 
        if (state.is_knot[i] == 1) {
          state.conv_cur[i] = (1-lambda) * state.conv_cur[i] + lambda * state.conv_new[i];
          if (state.conv_cur[i] > 0) {
            state.conv_cur[i] = 0;
          }
        }
        else {
          state.conv_cur[i] = 0;
        }
      }
      if (state.sl == 1) {
        state.phi_cur_slr[0] = (1-lambda) * state.phi_cur_slr[0] + lambda * state.phi_new_slr[0];
      }
      /* probably not really needed, but for diagnostic purposes (sees to it that slope is 
	 at all times correct */
   
      localmle_slope(&state);
    }  /* end of inner while loop */  

    for (i = 0; i < n; i++) {
      state.phi_cur[i] = state.phi_new[i];
      state.conv_cur[i] = state.conv_new[i];
    }
    state.phi_cur_slr[0] = state.phi_new_slr[0];
    j = 0;
    for (i = 0; i < n; i++) {
      if (state.is_knot[i] != state.is_knot_old[i]) {
        j = 1;
        break;
      } 
    }
    if (j == 0) {
      break;
    }
    /* next 4 lines: for testing the while condition in the next pass of while loop */
    for (i = 0; i < n; i++) {
      sabsH += fabs(state.H[i]);
    }
    maxH = amaxplus(n, state.H, &argmax); 

  }  /* end of outer while loop */
  Fhat[0] = 0;
  for (i = 1; i < n; i++) {
    Fhat[i] = Fhat[i-1] + state.dx[i-1] * J00(state.phi_cur[i-1], state.phi_cur[i]);
  }
  *Fhatfin = Fhat[n-1] + exp(state.phi_cur[n-1])/(- state.phi_cur_slr[0]);
  *L = state.L;  /* output for L (other output is always written at the right address,
                    this is not done for L (for various reasons, but timewise nothing really is lost by this
                    step because otherwise we have to define state.L as pointer and set it to the address
                    of L in the beginning). */
}



void localmle_slope(State *state)
/* implicit output: phi_new, phi_new_slr, L, conv_new, and H */ 
{
  int i,n,rn;
  int ind;  /* for the update of H */
  /*  int rknot;           new in slope version: index of last knot */
  double dtmp, sl2tmp, sumtmp, csumtmp, csum2wtmp, jtmp1, jtmp2;  /* for the update of H */
  double *xtmp, *wtmp;   /* for the update of H */
  RState rs;

  n = state->n;   /* remember state->n is short for (*state).n  */
  xtmp = state->xtmp;
  wtmp = state->w;      /* substitutes for rs->{thingy} */
  rs.p0 = state->p0;
  rn = state->rn;

  rs.knotlist = (int *) Calloc((long) rn, int);   /* "R-function", more flexible than R_alloc */
  rs.rx = (double *) Calloc((long) rn, double);   
  rs.rdx = (double *) Calloc((long) rn, double);
  rs.rw = (double *) Calloc((long) rn, double); 
  rs.rw_slr = (double *) Calloc((long) 1, double); /* hmm, this seems necessary */ 
  rs.rphi_cur = (double *) Calloc((long) rn, double);
  rs.rphi_cur_slr = (double *) Calloc((long) 1, double);
  rs.rphi_new = (double *) Calloc((long) rn, double);
  /* rs.rphi_new_slr = (double *) Calloc((long) 1, double); */
  rs.rgrad = (double *) Calloc((long) rn, double);
  rs.rmhess_diag = (double *) Calloc((long) rn, double);
  rs.rmhess_sub = (double *) Calloc((long) rn, double);
  rs.temp = (double *) Calloc((long) rn, double); 
  rs.b = (double *) Calloc((long) rn, double);
  
  LocalReduce_slope(state, &rs);
    /* initializes reduced state based on state; more precisely: 
       initializes rs.rx, rs.rw_slr, rs.rdx, rs.knotlist, rs.rphi_cur, rs.rphi_cur_slr  
       based on state->x, state->w, state->w_slr, state->is_knot, state->phi_cur, state->phi_cur_slr */

  mle_slope(&rs);
    /* updates rs.rphi_cur, rs.rphi_cur_slr and rs.L (rs.L is only needed in lower levels than localmle:
       state->L is *never* needed but simply copied from rs.L in the end; both Ls
       are just "carried through" so that we can report them (more exactly state->L in the end) 
       mle uses rs's .rphi_new, .rphi_new_slr, .rgrad, .rmhess_diag and .rmhess_sub as scratch space; 
       note in particular that the .rmhess vectors do most of the time not contain
       the corresponding diagonals of the Hessian, but some diagnostic vector of a LAPACK routine */

  LocalExtend_slope(&rs, state);
    /* translates reduced state back to state; more precisely:
       updates state->phi_new and state->phi_new_slr 
       based on state->x, state->is_knot, rs.rx, rs.rphi_cur, and rs.rphi_cur_slr */

  LocalConvexity_slope(&rs, state);
    /* updates state->conv_new
       based on state->x, state->n, state->phi_new, state->phi_new_slr, state->is_knot,
                rs->rphi_cur, rs->rphi_cur_slr and rs->rdx */
  
  /* compute new vector H of directional derivatives */
  /* NOTE: this is not as described in DHR08, we go in the direction of ---/\----functions,
     this way we don't have "to sum so much" (I think the true reason was more complicated);
     note that for epsilon small enough we can add epsilon times such a function to a concave
     function without losing concavity */
  for (i = 0; i < rn-1; i++) {
    state->H[rs.knotlist[i]] = 0;
    if (rs.knotlist[i+1] > rs.knotlist[i] + 1) {
      dtmp = rs.rdx[i];
      sumtmp = 0;   /* compute sum(wtmp * (1 - xtmp)) */
      for (ind = rs.knotlist[i] + 1; ind < rs.knotlist[i+1]; ind++) {
        xtmp[ind] = (state->x[ind] - state->x[rs.knotlist[i]])/dtmp;  
	  /* note xtmp at knot positions has arbitrary values (from earlier passes)
             put these values are not used */ 
        sumtmp += wtmp[ind] * (1 - xtmp[ind]);
      }
      csumtmp = csum2wtmp = 0;
      for (ind = rs.knotlist[i] + 1; ind < rs.knotlist[i+1]; ind++) {
        csumtmp += wtmp[ind] * xtmp[ind];
        csum2wtmp += wtmp[ind];
        state->H[ind] = dtmp * (csumtmp - xtmp[ind] * csum2wtmp + xtmp[ind] * sumtmp);
        jtmp1 = xtmp[ind] * J10(state->phi_new[ind], state->phi_new[rs.knotlist[i]]);
        jtmp2 = (1 - xtmp[ind]) * J10(state->phi_new[ind], state->phi_new[rs.knotlist[i+1]]);
        state->H[ind] = state->H[ind] - dtmp * dtmp * xtmp[ind] * (1 - xtmp[ind]) * (jtmp1 + jtmp2);
      }
    }
  }
  state->H[rs.knotlist[rn-1]] = 0; 
  /* If rs.sl==0, then rs.knotlist[rn-1] == n-1 ALWAYS! */
  if (n-1 > rs.knotlist[rn-1]) {  
    dtmp = rs.rdx[rn-1];  /* bringt wohl auch nicht mehr viel */
    sl2tmp = state->phi_new_slr[0] * state->phi_new_slr[0];
    csumtmp = csum2wtmp = 0;
    state->H[n-1] = - state->w_slr[0] + exp(state->phi_new[n-1]) / sl2tmp;
    for (ind = n-2; ind > rs.knotlist[rn-1]; ind--) {
      csum2wtmp += wtmp[ind+1];
      csumtmp += csum2wtmp * ((state->x[ind+1] - state->x[ind]) / dtmp); 
      state->H[ind] = - dtmp * csumtmp - state->w_slr[0] + exp(state->phi_new[ind]) / sl2tmp;
    }
  }  

  state->L = rs.L;

  Free(rs.knotlist);    /* the R-version of free */
  Free(rs.rx); 
  Free(rs.rdx);
  Free(rs.rw);
  Free(rs.rw_slr);
  Free(rs.rphi_cur);
  Free(rs.rphi_cur_slr);
  Free(rs.rphi_new);
  /*  Free(rs.rphi_new_slr); */
  Free(rs.rgrad);
  Free(rs.rmhess_diag);
  Free(rs.rmhess_sub);
  Free(rs.temp);
  Free(rs.b);
}



void mle_slope(RState *rs)
/* implicit outputs: rs->phi_cur, rs.rphi_cur_slr and rs->L */
/* This is the actual Newton-optimisation given the inactive constraints and on the reduced objects */
/* The philosophy is as follows: we basically maximize the _profile_ functional \tilde{L}(\psi) from DHR08, p.22
   (max over slopes) by Newton-steps; we make the Newton-step based on this function, but normalize before each
   step the current function phi_cur (and update L=\tilde{L}(\psi) for this normalized phi), which probably
   might make the L-value slightly lower for numerical reasons in particular cases. Note that we always
   (even if outer loop is never entered) end with a normalized function (integral of exp(phi) exactly 1)
   that may not have quite the optimal slope (according to the formula on p.22), but is very close.
      If we wanted to normalize in such a way that both conditions are satisfied exactly we would
   have to do another iteration procedure I think since the optimal slope depends on psi_m. 
   NEW VERSION (from 0.9): we do not keep track of the slope but assume that always the optimal one is
   used and explicitly rphi_cur_slr to that value only in the very end of this procedure. */
{
  int i,iter0 = 0, iter1, rn;
  double dirderiv;
  double sqrtw_slr;
  double L, L_new, tstar; /* rs->L from previous phi is not used of course (but updated) */
  double *rphi_cur, *rphi_new; /* ttt *rphi_cur_slr, *rphi_new_slr;  substitutes for rs->{thingy} */
  /* remember: rphi_cur_slr and rphi_new_slr have length 1 */    

  rn = rs->rn;
  rphi_cur = rs->rphi_cur;  /* this contains the info */
  rphi_new = rs->rphi_new;  /* this is scratch space */
  /* ttt rphi_cur_slr = rs->rphi_cur_slr;  this contains the info */
  /* ttt rphi_new_slr = rs->rphi_new_slr;  this is scratch space */

  sqrtw_slr = sqrt(rs->rw_slr[0]);  /* saves some computations */ 

  LocalNormalize_slope(rs);  /* operates on rphi_cur and rphi_cur_slr */
  /* the above line seems like a little overkill; normalization is performed by caller (first pass: in logcon
     for the full phi; later passes in this function when phi_cur is updated). However it seems to correct
     slight numerical errors (I suspect they stem from projecting the optimised function back on the cone).
     The errors seem to be of order around 10^(-5) per position, so I leave the normalisation in  */

  L = Local_LL_slope(rn, rs->rx, rs->rdx, rs->rw, rs->rw_slr, rphi_cur); 
  dirderiv = Local_LL_rest_slope(rs); 
        /* side effect: rs->rphi_new and rs->rphi_new_slr are updated */

  /* dirderiv is always the directional derivative at rphi_cur along the vector (rphi_new - rphi_cur)
     it is a more reliable way (I suppose) to measure whether newton steps still do someting (than
     say just the difference in L or the directional derivative along a unit vector) */
  while ((dirderiv >= PRECNEWTON) & (iter0 < 100)) {
    iter0++;
    L_new = Local_LL_slope(rn, rs->rx, rs->rdx, rs->rw, rs->rw_slr, rphi_new);
    iter1 = 0;
    /* Stepsize correction (Hermite approximation?) */
    while ((L_new < L) & (iter1 < 20)) {
      iter1++;
      for (i = 0; i < rn; i++) {
        rphi_new[i] = 0.5 * (rphi_cur[i] + rphi_new[i]);
      }
      /* ttt rphi_new_slr[0] = - exp(rphi_new[rn-1]/2) / sqrtw_slr; */
      /*   BETTER NOT rphi_new_slr[0] = 0.5 * (rphi_cur_slr[0] + rphi_new_slr[0]);
	   because otherwise our slope does not match the likelihood value produced */
      L_new = Local_LL_slope(rn, rs->rx, rs->rdx, rs->rw, rs->rw_slr, rphi_new);
      dirderiv = 0.5 * dirderiv;  /* remains true for slope version */
    }
    if (L_new >= L) {
      tstar = (L_new - L)/dirderiv;
      /* there might be a problem because of division by a very small number ("double overflow")
         but I rather don't think so: a number that is (intuitively) between 10^(-20) and 10
         divided by a number > 10^(-30) should be representable, and if the result is clearly bigger than
         0.5 it can be very imprecise and we don't care */ 
      if (tstar >= 0.5) {
        for (i = 0; i < rn; i++) {
          rphi_cur[i] = rphi_new[i];
        }
        /* ttt rphi_cur_slr[0] = rphi_new_slr[0]; */
        LocalNormalize_slope(rs);  /* operates on rs->rphi_cur and 
                                      slope version: and rs->rphi_cur_slr */
      }
      else {    /* seems to be a trick to get a better next value in Newton step (is this Hermite approx.?) */
        tstar = 0.5/(1 - tstar);
	for (i = 0; i < rn; i++) {
	  rphi_cur[i] = (1 - tstar) * rphi_cur[i] + tstar * rphi_new[i];
        }
        /* ttt rphi_cur_slr[0] = - exp(rphi_cur[rn-1]/2) / sqrtw_slr; */
	LocalNormalize_slope(rs); /* operates on rs->rphi_cur and rs->rphi_cur_slr */
      }
      L = Local_LL_slope(rn, rs->rx, rs->rdx, rs->rw, rs->rw_slr, rphi_cur);
      dirderiv = Local_LL_rest_slope(rs);  /* side effect: rs->rphi_new and rs->rphi_new_slr are updated */
      /* rs->rphi_new is needed for divderiv, which in turn is needed for checking the loop condition */
    }
    else {
      dirderiv = 0;
      warning("Likelihood decreased in Newton method after ssc; Newton stopped!");
    }
  }
  /* should work just as well without the if (rs->rphi_cur_slr[0] should stay put at -Inf
     if rs->sl is zero */
  if (rs->sl == 1) {
    rs->rphi_cur_slr[0] = - exp(rphi_cur[rn-1]/2) / sqrtw_slr;
  }
  rs->L = L; 
  /* everything is perfect here: especially, we *do* use the correct L  */
}




/* ------------ Helper functions ------------------------- */

/* z denotes the default "array output" in the following functions */
double J00(double r, double s)
{
  double d, z;
  /* we do without the fudge constant v=1, to speed things up */
  
  z = exp(r);  /* since we use it twice below; I think this is a tad faster */
  d = s - r;
  if (fabs(d) > 0.005) {
    z = z * (exp(d)-1.0)/d;
  }
  else {
    /* Horner scheme: faster, because only three multiplications instead of seven
       (three divisions for both methods, but 1/ might even be faster), and
       more precise because smaller numbers are added first */
    z = z * 
        (1.0 + 
          d * (0.5 +
            d * (1/6.0 + 
              d * (1/24.0 + d/120.0))));
  }  
  return(z);
}



double J10(double r, double s)
{
  double d, z;

  z = exp(r);
  d = s - r;
  if (fabs(d) > 0.01) {
    z = z * (exp(d)-1-d)/(d * d);
  }
  else {
    z = z * 
        (0.5 +
	 d * (1/6.0 + 
	   d * (1/24.0 +
	     d * (1/120.0 + d/720.0))));
  }  
  return(z);
}



double J11(double r, double s)
{
  double d, z;

  z = exp(r);
  d = s - r;
  if (fabs(d) > 0.02) {
    z = z * (d * (exp(d) + 1) - 2 * (exp(d)-1)) / (d * d * d);
  }
  else {
    z = z * 
        (1/6.0 + 
	  d * (1/12.0 +
            d * (1/40.0 + 
	      d * (1/180.0 + d/1008.0))));
  }  
  return(z);
}



double J20(double r, double s)
{
  double d, z;

  z = exp(r);
  d = s - r;
  if (fabs(d) > 0.02) {
    z = 2.0 * z * (exp(d) - 1.0 - d - d * d/2.0)/(d * d * d);
  }
  else {
    z = z * 
        (1/3.0 + 
	 d * (1/12.0 + 
	   d * (1/60.0 + 
	     d * (1/360.0 + d/2520.0))));
  }  
  return(z);
}



/* rphi is _cur or _new depending from where Local_LL is called */
double Local_LL_slope(int rn, double *rx, double *rdx, double *rw, double *rw_slr, double *rphi)
{
  int i;
  double sqrtw_slr, expphi;  
  double L;
  
  /* Here and elsewhere: if we do not use a slope (i.e. rs->sl=0, rs->rw_slr=0, rs->rphi_cur_slr=-Inf)
     the computations are still done (exactly!) the same way as in older versions (with no slope at all)
     even if there is no "if (rs->sl == 1)"-loop */
  sqrtw_slr = sqrt(rw_slr[0]);
  expphi = exp(rphi[rn-1]/2);

  L = 0;
  for (i = 0; i < rn-1; i++) {
    L += rw[i] * rphi[i] - rdx[i] * J00(rphi[i], rphi[i+1]); 
  } 
  L += rw[rn-1] * rphi[rn-1] - 2 * sqrtw_slr * expphi;
  return(L);
}



double Local_LL_rest_slope(RState *rs)
  /* implicit output: rs->rphi_new, rs->rphi_new_slr */
{
  int i,info = 0,rn;
  double s, sqrtw_slr, expphi;  /* temporary scalar variable */
  double dirderiv;  /* explicit output (directional derivative returned by this function) */
  double *rw, *rw_slr, *rdx, *rphi;  /* substitutes for rs->{thingy} */
  /*      double *temp;  used for the computation of mhess          */
  /* int *ipiv; used for inverting mhess */
  /*      double *b; used for inverting mhess; gradient at first then storing result       */  
    
  rn = rs->rn;
  rw = rs->rw;
  rw_slr = rs->rw_slr;
  rdx = rs->rdx;
  rphi = rs->rphi_cur;

  sqrtw_slr = sqrt(rw_slr[0]);  /* carefull: rw_slr is pointer, sqrtw_slr is not!! */
  expphi = exp(rphi[rn-1]/2);

  /* This solves a nasty memory leak problem due to the indices rn-2 */
  if (rn > 1) {
    rs->rgrad[0] = rw[0] - rdx[0] * J10(rphi[0],rphi[1]);
    for (i = 1; i < rn-1; i++) {
      rs->rgrad[i] = rw[i] - rdx[i] * J10(rphi[i],rphi[i+1]) - rdx[i-1] * J10(rphi[i],rphi[i-1]);
    } 
    rs->rgrad[rn-1] = rw[rn-1] - rdx[rn-2] * J10(rphi[rn-1],rphi[rn-2]) - sqrtw_slr * expphi;
  
    rs->temp[0] = rdx[0] * J20(rphi[0],rphi[1]);
    s = rs->temp[0];
    for (i = 1; i < rn-1; i++) {
      rs->temp[i] = rdx[i] * J20(rphi[i],rphi[i+1]) + rdx[i-1] * J20(rphi[i],rphi[i-1]);
      s += rs->temp[i];
    }
    rs->temp[rn-1] = rdx[rn-2] * J20(rphi[rn-1],rphi[rn-2]);
    s = (s + rs->temp[rn-1])/rn * 1.0e-12;
  
    for (i = 0; i < rn-1; i++) {
      rs->rmhess_diag[i] = rs->temp[i] + s;
      rs->rmhess_sub[i] = rdx[i] * J11(rphi[i],rphi[i+1]);
    }
    rs->rmhess_diag[rn-1] = rs->temp[rn-1] + s + 0.5 * sqrtw_slr * expphi;
    /* the +s solves numerical problems if rmhess is badly conditioned otherwise (I think) */
  }
  else {
    rs->rgrad[0] = rw[0] - sqrtw_slr * expphi;
    rs->rmhess_diag[0] = 0.5 * sqrtw_slr * expphi;
  }

  /* copy rgrad to b to */
  for (i=0; i<rn; i++) {
    rs->b[i] = rs->rgrad[i];
  }
  i = 1;   /* not very elegant, but we need a pointer to a 1 in the F77_CALL below */

  /* note that mhess should be coded as column major, but ours is row mojor;
     however since the matrix mhess is always symmetric it doesn't matter   */
  /* F77_CALL(dgesv)(&rn, &rn, rs->rmhess, &rn, ipiv, b, &rn, &info); */
  F77_CALL(dptsv)(&rn, &i, rs->rmhess_diag, rs->rmhess_sub, rs->b, &rn, &info); 
  /* replaced the all purpose solver dgesv from LAPACK by dptsv (also LAPACK);
     it is specialised for symmetric pos. definite tridiagonal matrices (i.e. ideal for our case);
     there seems to be roughly a 20% saving in total computation time for logcon from this,
     the fact that we can enter the (two!) diagonals directly and the fact that we solve 
     directly the equation system with rs->rgrad on the rhs. */
  /* THIS WORKS even in the slope version, where rn may be 1 (note also that rs->rmhess_sub has
     length rn not rn-1 and was initialized to 0 by Calloc) */
  if (info != 0) {
    error("Unable to invert matrix mhess in Local_LL_rest");
  }  

  dirderiv = 0;
  for (i = 0; i < rn; i++) {
    rs->rphi_new[i] = rphi[i] + rs->b[i];
    dirderiv += rs->rgrad[i] * (rs->rphi_new[i]-rphi[i]);
  }                

  /* ttt rs->rphi_new_slr[0] = - expphi / sqrtw_slr; */

  return(dirderiv);
}


/* Carefull: this is no longer true normalization based on rs->rphi_cur
   with value of rs->rphi_cur_rsl[0] as recorded,
   but based on rs->rphi_cur with optimal slope according to DHR, p.22
   The reason for this is that LocalNormalize_slope is only called from within mle_slope and
   there rs->rphi_cur_rsl[0] is never actually updated nor used but just the optimal term and then 
   rs->rphi_cur_rsl[0] is updated at the end of mle_slope */
void LocalNormalize_slope(RState *rs)
  /* implicit output written at rs->rphi_cur */
{
  int i,rn;
  double s = 0;

  rn = rs->rn;

  for (i = 0; i < rn-1; i++) {
    s += rs->rdx[i] * J00(rs->rphi_cur[i], rs->rphi_cur[i+1]);
  } 
  s += sqrt(rs->rw_slr[0]) * exp(rs->rphi_cur[rn-1]/2);
  for (i = 0; i < rn; i++) {
    rs->rphi_cur[i] += log(1.0 - rs->p0) - log(s); 
  }
 
}



 /* Initializes rs */
void LocalReduce_slope(State *state, RState *rs) 
  /* implicit output written at rs->rx, rs->rw, rs->rw_slr, rs->rdx, rs->knotlist,
     rs->rphi_cur and rs->rphi_cur_slr   */
{
  int i,j,ind,n,rn;
  double lambda, sum1tmp, sum2tmp;

  n = state->n;
  rn = rs->rn = state->rn;
  rs->sl = state->sl;

  j = 0;
  for (i = 0; i < n; i++) {
    if (state->is_knot[i] == 1) {
      rs->knotlist[j] = i;
      rs->rx[j] = state->x[i];
      rs->rw[j] = state->w[i];
      rs->rphi_cur[j] = state->phi_cur[i];
      j++;
    }
  }
  rs->rw_slr[0] = state->w_slr[0];
  rs->rphi_cur_slr[0] = state->phi_cur_slr[0];  

  if (j != rn) warning("This was knot to be expected! :-)  (in LocalReduce)  %d  %d", j, rn);
  for (j = 0; j < rn-1; j++) {
    rs->rdx[j] = rs->rx[j+1]-rs->rx[j];
    if (rs->knotlist[j+1] > rs->knotlist[j] + 1) {
      sum1tmp = 0;   /* for computing sum(w * (1 - lambda)) */
      sum2tmp = 0;   /* for computing sum(w * lambda) */
      for (ind = rs->knotlist[j] + 1; ind < rs->knotlist[j+1]; ind++) {
        lambda = (state->x[ind] - rs->rx[j]) / rs->rdx[j];
	sum1tmp += state->w[ind] * (1 - lambda);
        sum2tmp += state->w[ind] * lambda;
      }
      rs->rw[j] += sum1tmp;
      rs->rw[j+1] += sum2tmp;
    }
  }
  sum1tmp = 0;
  sum2tmp = 0;
  for (ind = rs->knotlist[rn-1] + 1; ind < n; ind++) {
    sum1tmp += state->w[ind];
    sum2tmp += state->w[ind] * (state->x[ind] - rs->rx[rn-1]);    
  }
  rs->rw[rn-1] += sum1tmp;
  rs->rw_slr[0] += sum2tmp;
  /* still true for slope version?: sum of rw is extremely close to 1 even for many points (large n at least)
     so we don't do an extra normalisation here */

  rs->rdx[rn-1] = state->x[n-1] - rs->rx[rn-1];
  /* slope version: zero if final x-point is a knot, dist last knot to final x-point in general */
} 



void LocalExtend_slope(RState *rs, State *state)
  /* implicit output written at state->phi_new */
{
  int j,ind,n,rn;
  double lambda;

  n = state->n;
  rn = rs->rn;
    
  for (j = 0; j < rn-1; j++) {
    state->phi_new[rs->knotlist[j]] = rs->rphi_cur[j];
    /* "if" is not needed (3 more occurrences in the code), but there is hardly any comp. gain from omitting) */
    if (rs->knotlist[j+1] > rs->knotlist[j] + 1) {   
      for (ind = rs->knotlist[j] + 1; ind < rs->knotlist[j+1]; ind++) {
        lambda = (state->x[ind] - rs->rx[j]) / rs->rdx[j];
        state->phi_new[ind] = (1 - lambda) * rs->rphi_cur[j] + lambda * rs->rphi_cur[j+1];
      }
    }
  }
  state->phi_new[rs->knotlist[rn-1]] = rs->rphi_cur[rn-1];
  /* note: if rs->sl==0 then rs->knotlist[rn-1]=n-1 always and loops of the type below are not even entered */
  for (ind = rs->knotlist[rn-1] + 1; ind < n; ind++) {
    state->phi_new[ind] = rs->rphi_cur[rn-1] + rs->rphi_cur_slr[0] * (state->x[ind] - rs->rx[rn-1]);
  }
  /* this is necessary, I think, because otherwise state->phi_new_slr[0] *never* gets the value -Inf */
  state->phi_new_slr[0] = rs->rphi_cur_slr[0];
}



/* LocalConvexity is now only called from localmle
   (in loccon we make appropriate computations directly),  */
void LocalConvexity_slope(RState *rs, State *state)
  /* implicit output written at state->conv_new */
{
  int i,j,n,rn;
  int lastknot;
  double deriv1, deriv2;
  
  j = 0;
  n = state->n;
  rn = rs->rn;
  lastknot = rs->knotlist[rn-1]; /* lastknot is always n-1 if rs->sl == 0 */
  state->conv_new[0] = 0;
  if (rn > 1) {
    deriv1 = (rs->rphi_cur[1] - rs->rphi_cur[0]) / rs->rdx[0];
    for (i = 1; i < lastknot; i++) {
      if (state->is_knot[i] == 1) {
        j++;
        deriv2 = (rs->rphi_cur[j+1] - rs->rphi_cur[j]) / rs->rdx[j];
        state->conv_new[i] = deriv2 - deriv1;
        deriv1 = deriv2;
      }
      else {
        state->conv_new[i] = 0;
      }
    }
    if (rs->sl == 1) {
      state->conv_new[lastknot] = rs->rphi_cur_slr[0] - deriv1;
    }
    else {
      /* if we don't separate the two cases this is =-Inf, 
         which probably leads to problems (we take amax of the vector!) */
      state->conv_new[lastknot] = 0; 
    }
  }
  /* if rn == 1 then lastknot = 0 ALWAYS */
  for (i = lastknot + 1; i < n; i++) {
    state->conv_new[i] = 0;
  }
}  



/* ------------ "Little Helpers" ------------------------- */

/* void diff(int n, double *x, double *dx)
{
  for (i = 0; i < n-1; i++) {
    dx[i] = x[i+1] - x[i];
  }
}   */

/* Maximal element of an array of type double */
double amax(int n, double *a)
{
  int i;
  double res;

  res = a[0];
  for(i = 1; i < n; i++) {
    if(a[i] > res) {
      res = a[i];
    }
  }
  return(res);
}

/* Maximal element of an array of type double, implicitly returns maximising index at loc */
double amaxplus(int n, double *a, int *loc)
{
  int i;
  double res;

  res = a[0];
  *loc = 0;
  for(i = 1; i < n; i++) {
    if(a[i] > res) {
      res = a[i];
      *loc = i;
    }
  }
  return(res);
}

/* Maximal element of the ABSOLUTE VALUES of an array of type double */
double amaxabs(int n, double *a)
{
  int i;
  double tmp,res;

  res = fabs(a[0]);
  for(i = 1; i < n; i++) {
    tmp = fabs(a[i]);
    if(tmp > res) {
      res = tmp;
    }
  }
  return(res);
}
