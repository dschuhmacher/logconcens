\name{plotint}
\alias{plotint}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Plot Censored Data
}
\description{
  Plot a graphical representation of censored data specified by a two-column matrix of left and right interval endpoints. The grid of potential knots used by \code{logcon} is also shown.
}
\usage{
plotint(x, knot.prec = IQR(x[x<Inf])/75, imarks = NULL) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
    a two-column matrix of left and right endpoints of data intervals.
}
  \item{knot.prec}{
    the maximal distance between two consecutive grid points in the depiction of the grid used by
    \code{logcon}.
}
  \item{imarks}{
    an optional vector of \dQuote{spots} to be marked by \sQuote{x} for the intervals.
}
}
\value{
Used for the side effect.
}
\author{
Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
Kaspar Rufibach \email{kaspar.rufibach@gmail.com}\cr
Lutz Duembgen \email{duembgen@stat.unibe.ch}
}

\seealso{
  \code{\link{plot.lcdensity}}
}
\examples{
## See the examples for logcon
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
