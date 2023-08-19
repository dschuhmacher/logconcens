\name{logconcens-package}
\alias{logconcens-package}
\alias{logconcens}
\docType{package}
\title{
Maximum Likelihood Estimation of a Log-Concave Density Based on Censored Data 
}
\description{
Based on independent intervals \eqn{X_i = [L_i,R_i]}, where \eqn{-\infty < L_i \leq R_i \leq \infty}{-Inf < L_i <= R_i <= Inf}, compute the maximum likelihood estimator of a (sub)probability density under the assumption that it is log-concave. For further information see Duembgen, Rufibach, and Schuhmacher (2013).
}
\details{
The main function is \code{\link{logcon}}, which offers computation of the MLE for many types of censored and also exact data. Various parameters can be set that allow for fine control of the underlying EM algorithm in \dQuote{difficult} situations. An object of type \code{lcdensity} is returned, for which \code{\link[=plot.lcdensity]{plot}}, \code{\link[=print.lcdensity]{print}}, and \code{\link[=summary.lcdensity]{summary}} methods are available. There is also a function \code{\link{loglike}} for computing the log-likelihood of a \code{lcdensity} object.  
}
\author{
Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
Kaspar Rufibach \email{kaspar.rufibach@gmail.com}\cr
Lutz Duembgen \email{duembgen@stat.unibe.ch}

Maintainer: Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
}
\references{
Duembgen, L., Rufibach, K., and Schuhmacher, D. (2014). Maximum-likelihood estimation of a log-concave density based on censored data. Electronic Journal of Statistics, 8(1), 1405-1437. \doi{10.1214/14-EJS930}
}
\keyword{package}
\examples{
## Simple examples with simulated data.
## For more detailed examples see the help for the function logcon.

## exact data
set.seed(10)
x <- rnorm(100)
res <- logcon(x)
\dontrun{
plot(res)
xi <- seq(-3,3,0.05)
lines(xi,log(dnorm(xi)))}

## interval censored data
x <- rgamma(50,3,1)
x <- cbind(x,x+rexp(50,1))
plotint(x)
res <- logcon(x)
\dontrun{plot(res, type="CDF")}

## right censored data with mass at infinity
set.seed(11)
x <- rgamma(50,3,1)
x <- cbind(x,ifelse(rexp(50,1/3) < x,Inf,x))
plotint(x)
res <- logcon(x,adapt.p0=TRUE)
\dontrun{plot(res, type="survival")}

## rounded/binned data
set.seed(12)
x <- round(rnorm(100))
x <- cbind(x-0.5,x+0.5)
plotint(x)
res <- logcon(x)
\dontrun{
plot(res)
xi <- seq(-3,3,0.05)
lines(xi,log(dnorm(xi)))}
}
