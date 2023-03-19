\name{lc.control}
\alias{lc.control}
\alias{control}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Set the Control Parameters for \code{logcon}
}
\description{
Allows to set the control parameters for the more technical aspects of the function \code{\link{logcon}} and provides default values for any parameters that are not set.
}
\usage{
lc.control(maxiter=49, move.prec=1e-5, domind1l=1, domind2r=1, force.inf=FALSE,
                red.thresh=NULL, check.red=TRUE, addpoints=FALSE, addeps=NULL,
                preweights=NULL, minw=0, show=FALSE, verbose=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{maxiter}{
    the maximal number of iterations in the main EM algorithm. Default is \code{49} rather than \code{50},
    because this goes well with plotting in case of \code{show = TRUE}. 
}
  \item{move.prec}{
    a real number giving the threshold for the \eqn{L_1}{L1}-distance between densities in subsequent steps below which the algorithm is stopped.
  }
  \item{domind1l, domind2r}{
    index numbers in the vector of sorted interval endpoints that specify the left and right boundary of the maximal domain to be considered by the algorithm; see the details section of the help for \code{logcon}. The indices are counted from the left and from the right, respectively. So the default values of \code{domind1l = 1} and \code{domind2r = 1} mean that the largest possible domain is used. 
}
  \item{force.inf}{
    \code{logical}. For experimental use only. Should the domain interval be forced to be right-infinite (if there is a right-infinite data interval)? 
}
  \item{red.thresh}{
    a real number indicating the threshold below which the boundary integrals are considered too small; see the details section of the help for \code{logcon}. There is a sensible default, which depends on \code{check.red}.
}
  \item{check.red}{
    \code{logical}. If a boundary integral is deemed too small, should the derivative of the augmented log-likelihood be checked to confirm the domain reduction. 
}
  \item{addpoints}{
    \code{logical}. Should extra exact observations be added to the data at the left- and rightmost finite interval endpoints to prevent domain reduction? These observations obtain a small weight \eqn{<1} as compared to the weight of 1 for all the other observation intervals. The weight is specified by \code{addeps}.
}
  \item{addeps}{
    a positive real number. If \code{NULL}, a default value of \eqn{1/n^2}{1/n^2} is computed where \eqn{n} is the number of observation intervals. See \code{addpoints}.
}
  \item{preweights}{
    a vector of weights for the observation intervals. Defaults to \code{rep(1,n)}.
}
  \item{minw}{
    a positive real number. This gives another way for preventing domain reduction. Instead of adding
    observations the weights for the internal active set algorithm are kept at or above minw at the boundary
    of the domain. 
  }
  \item{show}{
    \code{logical}. Should progress of the algorithm be plotted? Warning: if \code{TRUE}, this may open
    many new graphics devices in case of complicated data sets.
}
  \item{verbose}{
    \code{logical}. Should additional information about the progress of the algorithm be printed? 
    This mainly prints quantities important for the decision to reduce the domain of the function
    and about the progress of the EM algorithm.  
}
}
\details{
For further explanations about the algorithm see the help for \code{logcon}. In summary:

\code{maxiter} and \code{move.prec} provide stopping criteria for the EM algorithm.

\code{domind1l}, \code{domind2r}, \code{force.inf}, \code{red.thresh}, and \code{check.red} control aspects related to domain reduction.

\code{addpoints}, \code{addeps}, \code{preweights}, \code{winw} allow for reweighing of data interval, mainly for increasing numerical stability by preventing domain reduction.

\code{show} and \code{verbose} give illustrations and background information of the run of the algorithm.   
}
\value{
A list with all of the above components set to their (specified or default) value.  
}
%\references{}
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
## See the examples for logcon
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
