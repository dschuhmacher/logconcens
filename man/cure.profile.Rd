\name{cure.profile}
\alias{cure.profile}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Evaluate the Profile Log-Likelihood on a Grid of \eqn{p_0}-Values
}
\description{
For each of a series of values for the cure parameter \eqn{p_0}{p0} run the function \code{logcon} and evaluate the (normalized) log-likelihood at \eqn{(\phi,p_0)}{(phi,p_0)}, where \eqn{\phi}{phi} is the log subprobability density returned by \code{logcon}. This serves for (approximate) joint likelihood maximization in \eqn{(\phi,p_0)}{(phi,p0)}.
}
\usage{
cure.profile(x, p0grid=seq(0,0.95,0.05), knot.prec=IQR(x[x<Inf])/75,
                  reduce=TRUE, control=lc.control())
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
    a two-column matrix of \eqn{n \geq 2}{n >= 2} rows containing the data intervals.
  }
  \item{p0grid}{
    a vector of values \eqn{p_0} for which the profile log-likelihood is to be evaluated.
  }
  \item{knot.prec, reduce, control}{
    arguments passed to the function \code{\link{logcon}}.
  }
}
\note{
  For a large \code{p0grid}-vector (fine grid) computations may take a long time. Consider using the option \code{adapt.p0} in the function \code{logcon} for a much faster method of joint likelihood maximization in \eqn{(\phi,p_0)}{(phi,p0)}.
}

\value{
  A list containing the following values:
  \item{p0hat }{the element in \code{p0grid} that maximizes the profile likelihood (in the very unlikely case of ties, only the smallest such element is returned).}
\item{status }{the vector of (normalized) profile log-likelihood values for the elements of \code{p0grid}.} 
}
\author{
Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
Kaspar Rufibach \email{kaspar.rufibach@gmail.com}\cr
Lutz Duembgen \email{duembgen@stat.unibe.ch}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{logcon}}, \code{\link{loglike}}
}
\examples{
## The example from the logconcens-package help page:
set.seed(11)
x <- rgamma(50,3,1)
x <- cbind(x,ifelse(rexp(50,1/3) < x,Inf,x))

\dontrun{
plotint(x)
progrid <- seq(0.1,0.6,0.025)
prores <- cure.profile(x, progrid)
plot(progrid, prores$loglike)
prores$p0hat
res <- logcon(x, p0=prores$p0hat)
plot(res, type="survival")
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ likelihood }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
