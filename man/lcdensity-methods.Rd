\name{lcdensity-methods}
\alias{lcdensity-methods}
\alias{plot.lcdensity}
\alias{print.lcdensity}
\alias{summary.lcdensity}

\title{
Methods for Objects of Class \code{lcdensity}
}
\description{
Plot, print, and summary methods for objects of class \code{lcdensity}.
}
\usage{
\method{plot}{lcdensity}(x, type = c("log-density", "density", "CDF", "survival"), sloperange = TRUE,
                      kinklines=TRUE, kinkpoints=FALSE, xlim=NULL, ylim=NULL, ...)

\method{print}{lcdensity}(x, ...)

\method{summary}{lcdensity}(object, ...)
}
\arguments{
  \item{x, object}{objects of class \code{lcdensity}, as returned by \code{\link{logcon}}.}
  \item{type}{the type of plot to be produced.}
  \item{sloperange}{\code{logical}. In cases where the cure parameter / the right-hand side slope of the
        log-subdensity \eqn{\phi}{phi} is not unique, should grey area be drawn indicating the set of
        possible right-hand slopes?}
  \item{kinklines}{\code{logical}. Should vertical lines be drawn at the kinks of the log-subdensity \eqn{\phi}{phi}?}
  \item{kinkpoints}{\code{logical}. Should fat points be plotted at the kinks of the log-subdensity \eqn{\phi}{phi}?}
  \item{xlim, ylim}{numeric vectors of length 2, giving the x and y coordinates ranges.}
  \item{\dots}{further arguments passed to \code{\link{plot.default}}. Depending on the argument
    this may or may not work in the intended way.}
}
\author{
Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
Kaspar Rufibach \email{kaspar.rufibach@gmail.com}\cr
Lutz Duembgen \email{duembgen@stat.unibe.ch}
}
\seealso{\code{\link{plotint}}}
\examples{
## See the examples for logcon
}
