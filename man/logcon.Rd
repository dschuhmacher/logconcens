\name{logcon}
\alias{logcon}
\alias{logconcure}
\alias{logConCens}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Compute Log-Concave MLE Based on Censored or Exact Data
}
\description{
Based on independent intervals \eqn{X_i = [L_i,R_i]}, where \eqn{-\infty < L_i \leq R_i \leq \infty}{-Inf < L_i <= R_i <= Inf}, compute the maximum likelihood estimator of a (sub)probability density \eqn{\phi}{phi} and the remaining mass \eqn{p_0}{p0} at infinity (also known as \emph{cure parameter}) under the assumption that the former is log-concave. Computation is based on an EM algorithm. For further information see Duembgen, Rufibach, and Schuhmacher (2013, preprint).
}
\usage{
logcon(x, adapt.p0=FALSE, p0=0, knot.prec=IQR(x[x<Inf])/75, reduce=TRUE,
            control=lc.control())

logConCens(x, adapt.p0=FALSE, p0=0, knot.prec=IQR(x[x<Inf])/75, reduce=TRUE,
                control=lc.control())

logconcure(x, p0=0, knot.prec=IQR(x[x<Inf])/75, reduce=TRUE, control=lc.control())
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
    a two-column matrix of \eqn{n \geq 2}{n >= 2} rows containing the data intervals, or a vector of length
    \eqn{n \geq 2}{n >= 2} containing the exact data points.
}
  \item{adapt.p0}{
    \code{logical}. Should the algorithm be allowed to adapt \eqn{p_0}{p0}? In this case an alternating maximization procedure is used that is \emph{believed} to always yield a joint maximizer \eqn{(\hat{\phi},\hat{p_0})}{(phi,p0)}. For the much slower (but maybe safer) profile likelihood maximization method, see the function \code{\link{cure.profile}}.
}
  \item{p0}{
    a number from 0 to 1 specifying the mass at infinity. 
    If the algorithm is allowed to adapt \eqn{p_0}{p0}, this argument only specifies the starting value.
    Otherwise it is assumed that the true cure parameter \eqn{p_0}{p0} is equal to this number. In particular,
    for the default setting of 0, a proper probability density \eqn{\phi}{phi} is estimated.
}
  \item{knot.prec}{
    the maximal distance between two consecutive grid points, where knots (points at which the resulting
    log-subdensity \eqn{\phi}{phi} may change slope) can be positioned. See details.
}
  \item{reduce}{
    \code{logical}. Should the domain of the (sub)density be reduced whenever the mass at the left or the right boundary becomes too small?
}
  \item{control}{
    a list of control parameters for the more technical aspects of the algorithm; usually the result of a call
    to \code{\link{lc.control}}.
  }
}
\details{
Based on the data intervals \eqn{X_i = [L_i,R_i]} described above, function \code{logcon} computes a concave, piecewise linear function \eqn{\phi}{phi} and a probability \eqn{p_0}{p0} which satisfy \eqn{\int \exp \phi(x) \, dx = 1-p_0}{int exp(phi(x)) dx = 1-p0} and jointly maximize the (normalized) log-likelihood.

\deqn{\ell(\phi, p_0) = \frac{1}{n} \sum_{i=1}^n \biggl[ 1\{L_i = R_i\} \phi(X_i) + 1\{L_i < R_i\}
   \log \biggl( \int_{L_i}^{R_i} \exp \phi(x) \; dx + 1\{R_i = \infty\} p_0 \biggr) \ \biggr],}{%
   l(\phi, p_0) = (1/n) \sum_{i=1}^n  [  1{L_i = R_i} phi(X_i) + 1{L_i < R_i}
   log ( int_{L_i}^{R_i} exp phi(x) \, dx + 1{R_i = Inf} p_0 )  ].}

If \code{x} is a two-column matrix, it is assumed to contain the left and right interval endpoints in the correct order. Intervals may have length zero (both endpoints equal) or be unbounded to the right (right endpoint is \code{Inf}). Computation is based on an EM algorithm, where the M-step uses an active set algorithm for computing the log-concave MLE for exact data with weights. The active set algorithm was described in Duembgen, Huesler, and Rufibach (2007) and Duembgen and Rufibach (2011) and is available in the R package \code{logcondens}. It has been re-implemented in C for the current package because of speed requirements. The whole algorithm for censored data has been indicated in Duembgen, Huesler, and Rufibach (2007) and was elaborated in Duembgen, Schuhmacher, and Rufibach (2013, preprint).

If \code{x} is a vector argument, it is assumed to contain the exact data points. In this case the active set algorithm is accessed directly.

In order to obtain a finite dimensional optimization problem the (supposed) domain of \eqn{\phi}{phi} is subdivided by a grid. Stretches between interval endpoints where for theoretical reasons no knots (points where the slope of \eqn{\phi}{phi} changes) can lie are left out. The argument \code{kink.prec} gives the maximal distance we allow between consecutive grid points in stretches where knots can lie. Say \code{plotint(x)} to see the grid.

The EM algorithm works only for fixed dimensionality of the problem, but the domain of the function \eqn{\phi}{phi} is not a priori known. Therefore there is an outer loop starting with the largest possible domain, given by the minimal and maximal endpoints of all the intervals, and decreasing the domain as soon as the EM steps let \eqn{\phi}{phi} become very small towards the boundary. \dQuote{Very small} means that the integral of \eqn{\exp \circ \, \phi}{exp o phi} over the first or last stretch between interval endpoints within the current domain falls below a certain threshold \code{red.thresh}, which can be controlled via \code{\link{lc.control}}.

Domain reduction tends to be rather conservative. If the computed solution has a suspiciously steep slope at any of the domain boundaries, the recommended strategy is to enforce a smaller domain by increasing the parameters \code{domind1l} and/or \code{domind2r} via \code{\link{lc.control}}. The function \code{\link{loglike}} may be used to compare the (normalized) log-likelihoods of the results.

\code{logConCens} is  an alias for \code{logcon}. It is introduced to provide unified naming with the main functions in the packages \code{logcondens} and \code{logcondiscr}. 

\code{logconcure} is the same as \code{logcon} with \code{adapt.p0 = TRUE} fixed. 
%
%
}
\value{
An object of class \code{lcdensity} for which reasonable \code{\link[=plot.lcdensity]{plot}}, \code{\link[=print.lcdensity]{print}}, and \code{\link[=summary.lcdensity]{summary}} methods are available.

If the argument \code{x} is a two-column matrix (censored data case), such an object has the following components. 
\item{basedOn }{the string \code{"censored"} for the type of data the solution is based on.}
\item{status }{currently only \code{0} if the algorithm converged; and \code{1} otherwise. 
   Note that in most cases even with status \code{1} the returned solution is very close to the truth.
   The \code{1} is often due to the fact that the termination criterion is not so well balanced yet.}
\item{x }{the data entered.}
\item{tau }{the ordered vector of different interval endpoints.}
\item{domind1, domind2}{the indices of the \code{tau}-element at which the domain of the MLE \eqn{\phi}{phi}%
   starts/ends.}
\item{tplus}{the grid vector. \code{tau[domind1:domind2]} augmented by points of subdivision.}
\item{isKnot}{\code{0}-\code{1} value. For the finite elements of \code{tplus} a \code{1} if \eqn{\phi}{phi} has a knot at this position, \code{0} otherwise.}
\item{phi}{the vector of \eqn{\phi}{phi}-values at the finite elements of \code{tplus}.}
\item{phislr}{if \eqn{\sup({\rm dom}(\phi)) = \infty}{sup(dom(phi)) = Inf}, the slope of \eqn{\phi}{phi} after the last knot. Otherwise \eqn{-\infty}{-Inf}.}
\item{phislr.range}{a vector of length 2 specifying a range of possible values for \code{phislr}. This is for the (rather rare) situations that mass may be shifted between the interval from the rightmost tau-point to infinity and the cure parameter without changing the likelihood. Otherwise \code{phislr.range} is \code{NA}.}
\item{cure}{the cure parameter. Either the original argument \code{p0} if \code{adapt.p0} was \code{FALSE}, otheriwse the estimated cure parameter obtained by the alternating maximization procedure.}
\item{cure.range}{a vector of length 2 specifying a range of possible values for \code{cure} or \code{NA}. See \code{phislr.range}.}
\item{Fhat}{the vector of values of the distribution function \eqn{F} of \eqn{\exp \circ \, \phi}{exp o phi} at the finite elements of \code{tplus}.}
\item{Fhatfin}{the computed value of \eqn{\lim_{t \to \infty} F(t)}{lim_{t to Inf} F(t)}.}
}

\references{
Duembgen, L., Huesler, A., and Rufibach, K. (2007). Active set and EM algorithms for log-concave densities based on complete and censored data. Technical Report 61. IMSV, University of Bern. \url{https://arxiv.org/abs/0707.4643}

Duembgen, L. and Rufibach, K., (2011). logcondens: Computations Related to Univariate Log-Concave Density Estimation. Journal of Statistical Software, 39(6), 1-28. \doi{10.18637/jss.v039.i06}

Duembgen, L., Rufibach, K., and Schuhmacher, D. (2014). Maximum-likelihood estimation of a log-concave density based on censored data. Electronic Journal of Statistics, 8(1), 1405-1437. \doi{10.1214/14-EJS930}
}
\author{
Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
Kaspar Rufibach \email{kaspar.rufibach@gmail.com}\cr
Lutz Duembgen \email{duembgen@stat.unibe.ch}
}
\note{
 If \code{x} is a vector, this function does the same as the function \code{logConDens} in the package
 \code{logcondens}. The latter package offers additional features such as grid-based computation with weights
 (for high numerical stability) and
 smoothing of the estimator, as well as nicer plotting. \strong{For exact data} we recommend using
 \code{logConDens} for
 everyday data analysis. \code{logcon} with a vector argument is to be preferred if time is of the essence (for
 data sets with several thousands of points or repeated optimization in iterative algorithms) or
 if an additional slope functionality is required. 

 Two other helpful packages for log-concave density estimation based on exact data are \code{logcondiscr} for estimating a discrete distribution and \code{LogConcDEAD} for estimating a multivariate continuous distribution.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{lc.control}}, \code{\link{lcdensity-methods}}, \code{\link{loglike}}
}
\examples{
# A function for artificially censoring exact data
censor <- function(y, timemat) {
  tm <- cbind(0,timemat,Inf)
  n <- length(y)
  res <- sapply(1:n, function(i){
            return( c( max(tm[i,][tm[i,] < y[i]]), min(tm[i,][tm[i,] >= y[i]]) ) ) } )
  return(t(res))
}



# ------------------------
#  interval censored data
# ------------------------

set.seed(20)
n <- 100
# generate exact data:
y <- rgamma(n,3)
# generate matrix of inspection times:
itimes <- matrix(rexp(10*n),n,10)
itimes <- t(apply(itimes,1,cumsum))
# transform exact data to interval data
x <- censor(y, itimes)
# plot both
plotint(x, imarks=y)

# Compute censored log-concave MLE
# (assuming only the censored data is available to us)
res <- logcon(x)
plot(res)

# Compare it to the log-concave MLE for the exact data
# and to the true Gamma(3,1) log-density
res.ex <- logcon(y)
lines(res.ex$x, res.ex$phi, lwd=2.5, lty=2)
xi <- seq(0,14,0.05)
lines(xi,log(dgamma(xi,3,1)), col=3, lwd=2)



# -------------------------
#  censored data with cure
# -------------------------

\dontrun{
set.seed(21)
n <- 100
# generate exact data:
y <- rgamma(n,3)
cured <- as.logical(rbinom(n,1,0.3))
y[cured] <- Inf

# generate matrix of inspection times:
itimes <- matrix(rexp(6*n),n,6)
itimes <- t(apply(itimes,1,cumsum))
# transform exact data to interval data
x <- censor(y, itimes)
# plot both
plotint(x, imarks=y)

# Compute censored log-concave MLE including cure parameter
# (assuming only the censored data is available to us)
res <- logcon(x, adapt.p0=TRUE)
plot(res)
# There is a trade-off between right-hand slope and cure parameter here
# (seen by the grey area on the right), but the margin is very small:
res$cure.range

# Compare the corresponding CDF to the true CDF
plot(res, type="CDF")
xi <- seq(0,14,0.05)
lines(xi,0.7*pgamma(xi,3,1), col=3, lwd=2)
# Note that the trade-off for the right-hand slope is not visible anymore 
# (in terms of the CDF the effect is too small)
}



# ------------------------------------
#  real right censored data with cure
# ------------------------------------

# Look at data set ovarian from package survival
# Gives survival times in days for 26 patients with advanced ovarian carcinoma,
# ignoring the covariates

# Bring data to right format and plot it
\dontrun{
library(survival)
data(ovarian)
sobj <- Surv(ovarian$futime, ovarian$fustat)
x <- cbind(sobj[,1], ifelse(as.logical(sobj[,2]),sobj[,1],Inf))
plotint(x)

# Compute censored log-concave MLE including cure parameter
res <- logcon(x, adapt.p0=TRUE)

# Compare the corresponding survival function to the Kaplan-Meier estimator
plot(res, type="survival")
res.km <- survfit(sobj ~ 1)
lines(res.km, lwd=1.5)}



# ----------------------
#  current status data
# ----------------------

\dontrun{
set.seed(22)
n <- 200
# generate exact data
y <- rweibull(n,2,1)
# generate vector of inspection times
itime <- matrix(rexp(n),n,1)
# transform exact data to interval data
x <- censor(y, itime)
# plot both
plotint(x, imarks=y)

# Compute censored log-concave MLE
# (assuming only the censored data is available to us)
res <- logcon(x)
plot(res, type="CDF")

# Compare it to the true Weibull(2,1) c.d.f.
xi <- seq(0,3,0.05)
lines(xi,pweibull(xi,2,1), col=3, lwd=2)}



# ----------------------
#  rounded/binned data
# ----------------------

\dontrun{
set.seed(23)
n <- 100
# generate data in [0,1] rounded to one digit
y <- round(rbeta(n,2,3),1)
# bring data to right format and plot it
x <- cbind(y-0.05,y+0.05)
plotint(x)

# Compute censored log-concave MLE
res <- logcon(x)
plot(res, type="density", xlim=c(0,1))

# Compare it to the true Beta(2,3) density
xi <- seq(0,1,0.005)
lines(xi,dbeta(xi,2,3), col=3, lwd=2)
# The peaks in the estimated density are often considered unsatisfactory
# However, they are barely noticeable in the c.d.f.
plot(res, type="CDF", xlim=c(0,1))
lines(xi,pbeta(xi,2,3), col=3, lwd=2)

# To get rid of them in the density apply the smoothing
# proposed in the package logcondens (to be implemented here)}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
