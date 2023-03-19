\name{loglike}
\alias{loglike}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Compute Log-Likelihood for an Object of Class \code{lcdensity}
}
\description{
Compute the (normalized) log-likelihood for an object of class \code{lcdensity} as described in the details section for the function \code{\link{logcon}}. The main use of this function is for comparing different results from \code{logcon} based on different (starting) domains.
}
\usage{
loglike(lcd)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{lcd}{an object of class \code{lcdensity}}
}
\value{
  A single numeric value, the (normalized) log-likelihood.
}
\author{
Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
Kaspar Rufibach \email{kaspar.rufibach@gmail.com}\cr
Lutz Duembgen \email{duembgen@stat.unibe.ch}
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{logcon}}
}
\examples{
  x <- matrix(c(0,0.5,0.5,1,1,2,3,3),4,2)
  res <- logcon(x)
  loglike(res)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ likelihood }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line


%x <- matrix(c(0,0.5,1,2),2,2)
%res <- logcon(x, show=FALSE)
%\dontrun{plot(res)}
%## definitely not a very intuitive solution
%
%res2 <- logcon(x, domind1 = res$domind1+1, domind2 = res$domind2, show=FALSE)
%\dontrun{plot(res2)}
%## that's more like it; it is easy to see that this is a theoretically
%## optimal solution; as is *any other* density that is concentrated on [0.5,1] 
%
%loglike(res)
%loglike(res2)
%## confirmed by loglike
%## (so the programmers should improve on the slope reduction mechanism... ;-)
