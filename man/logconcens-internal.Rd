\name{logconcens-internal}
\alias{GetWeights}
\alias{J00}
\alias{J10}
\alias{clc.fixdom}
\alias{subdivisor}
%
\title{Internal logconcens Functions}
\description{Internal logconcens functions, usually not to be called by the user.}
\usage{
GetWeights(preweights,tplus, p0, xindex, rightinf, phi, phislr, needsl)
J00(x, y, v = 1)
J10(x, y)
clc.fixdom(x, preweights=rep(1,dim(x)[1]), minw=0,
   p0, adapt.p0 = FALSE, reduce=TRUE, red.thresh=NULL,
   check.red=TRUE, force.inf=FALSE, tau, subdivide, domind1, 
   domind2, maxiter=60, knot.prec=IQR(x[x<Inf])/75,
   move.prec=1e-5, show=TRUE, verbose=FALSE)
subdivisor(j, tau, eps = 0.01)
phidivisor(j, tau, phi, eps=0.01)
}
\author{
Dominic Schuhmacher \email{dominic.schuhmacher@mathematik.uni-goettingen.de}\cr
Kaspar Rufibach \email{kaspar.rufibach@gmail.com}\cr
Lutz Duembgen \email{duembgen@stat.unibe.ch}
}
\keyword{internal}

