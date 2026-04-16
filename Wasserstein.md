# Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations

Peyman Mohajerin Esfahani1 $\textcircled{1}$ ·Daniel Kuhn²

Received: 9 May 2015 / Accepted: 16 June 2017 /Published online: 7 July 2017 $^ ©$ The Author(s) 2O17. This article is an open access publication

Abstract We consider stochastic programs where the distribution of the uncertain parameters is only observable through a finite training dataset. Using the Wasserstein metric, we construct a ball in the space of (multivariate and non-discrete) probability distributions centered at the uniform distribution on the training samples,and we seek decisions that perform best in view of the worst-case distribution within this Wasserstein ball. The state-of-the-art methods for solving the resulting distributionally robust optimization problems rely on global optimization techniques, which quickly become computationally excruciating. In this paper we demonstrate that, under mild assumptions,the distributionally robust optimization problems over Wasserstein balls can in fact be reformulated as finite convex programs—in many interesting cases even as tractable linear programs. Leveraging recent measure concentration results， we also show that their solutions enjoy powerful finite-sample performance guarantees. Our theoretical results are exemplified in mean-risk portfolio optimization as well as uncertainty quantification.

Mathematics Subject Clasification 90C15 Stochastic programming $\cdot$ 90C25 Convex programming $\cdot$ 90C47 Minimax problems

# 1 Introduction

Stochastic programming is a powerful modeling paradigm for optimization under uncertainty. The goal of a generic single-stage stochastic program is to find a decision $x \in \mathbb { R } ^ { n }$ that minimizes an expected cost $\mathbb { E } ^ { \mathbb { P } _ { [ h ( x , \xi ) ] } }$ , where the expectation is taken with respect to the distribution $\mathbb { P }$ of the continuous random vector $\xi \in \mathbb { R } ^ { m }$ . However, classical stochastic programming is challenged by the large-scale decision problems encountered in today's increasingly interconnected world. First, the distribution $\mathbb { P }$ is never observable but must be inferred from data. However, if we calibrate a stochastic program to a given dataset and evaluate its optimal decision on a different dataset, then the resulting out-of-sample performance is often disappointing-even if the two datasets are generated from the same distribution. This phenomenon is termed the optimizer's curse and is reminiscent of overfitting effects in statistics [48]. Second, in order to evaluate the objective function of a stochastic program for a fixed decision $x$ ， we need to compute a multivariate integral, which is #P-hard even if $h ( x , \xi )$ constitutes the positive part of an affine function, while $\xi$ is uniformly distributed on the unit hypercube [24, Corollary 1].

Distributionally robust optimization is an alternative modeling paradigm, where the objective is to find a decision $x$ that minimizes the worst-case expected cost $\begin{array} { r } { \operatorname* { s u p } _ { \mathbb { Q } \in \mathcal { P } } \mathbb { E } ^ { \mathbb { Q } } [ h ( x , \xi ) ] } \end{array}$ . Here, the worst-case is taken over an ambiguity set $\mathcal { P }$ ，that is,a family of distributions characterized through certain known properties of the unknown data-generating distribution $\mathbb { P }$ . Distributionally robust optimization problems have been studied since Scarf's [43] seminal treatise on the ambiguity-averse newsvendor problem in 1958,but the field has gained thrust only with the advent of modern robust optimization techniques in the last decade [3,9]. Distributionally robust optimization has the following striking benefits. First, adopting a worst-case approach regularizes the optimization problem and thereby mitigates the optimizer's curse characteristic for stochastic programming. Second, distributionally robust models are often tractable even though the corresponding stochastic model with the true data-generating distribution (which is generically continuous） are #P-hard. So even if the data-generating distribution was known, the corresponding stochastic program could not be solved efficiently.

The ambiguity set $\mathcal { P }$ is a key ingredient of any distributionally robust optimization model. A good ambiguity set should be rich enough to contain the true data-generating distribution with high confidence. On the other hand, the ambiguity set should be small enough to exclude pathological distributions, which would incentivize overly conservative decisions. The ambiguity set should also be easy to parameterize from data,and—ideally—it should facilitate a tractable reformulation of the distributionally robust optimization problem as a structured mathematical program that can be solved with off-the-shelf optimization software.

Distributionally robust optimization models where $\xi$ has finitely many realizations are reviewed in [2,7,39]. This paper focuses on situations where $\xi$ can have a continuum of realizations. In this setting, the existing literature has studied three types of ambiguity sets. Moment ambiguity sets contain all distributions that satisfy certain moment constraints, see for example [18,22,51] or the references therein. An attractive alternative is to define the ambiguity set as a ballin the space of probability distributions by using a probability distance function such as the Prohorov metric [20], the Kullback-Leibler divergence [25,27], or the Wasserstein metric [38,52] etc. Such metric-based ambiguity sets contain all distributions that are close to a nominal or most likely distribution with respect to the prescribed probability metric. By adjusting the radius of the ambiguity set, the modeler can thus control the degree of conservatism of the underlying optimization problem. If the radius drops to zero,then the ambiguity set shrinks to a singleton that contains only the nominal distribution,in which case the distributionally robust problem reduces to an ambiguity-free stochastic program. In addition, ambiguity sets can also be defined as confidence regions of goodness-of-fit tests [7].

In this paper we study distributionally robust optimization problems with a Wasserstein ambiguity set centered at the uniform distribution $\widehat { \mathbb { P } } _ { N }$ on $N$ independent and identically distributed training samples. The Wasserstein distance of two distributions $\mathbb { Q } _ { 1 }$ and $\mathbb { Q } _ { 2 }$ can be viewed as the minimum transportation cost for moving the probability mass from $\mathbb { Q } _ { 1 }$ to $\mathbb { Q } _ { 2 }$ ,and the Wasserstein ambiguity set contains all (continuous or discrete) distributions that are sufciently close to the (discrete) empirical distribution ${ \widehat { \mathbb { P } } } _ { N }$ with respect to the Wasserstein metric.Modern measure concentration results from statistics guarantee that the unknown data-generating distribution $\mathbb { P }$ belongs to the Wasserstein ambiguity set around ${ \widehat { \mathbb { P } } } _ { N }$ with confidence $1 - \beta$ if its radius is a sublinearly growing function of $\log ( 1 / \beta ) / N$ [11,21]. The optimal value of the distributionally robust problem thus provides an upper confidence bound on the achievable out-of-sample cost.

While Wasserstein ambiguity sets offer powerful out-of-sample performance guarantees and enable the decision maker to control the model's conservativeness, moment-based ambiguity sets appear to display better tractability properties. Specifically, there is growing evidence that distributionally robust models with moment ambiguity sets are more tractable than the corresponding stochastic models because the intractable high-dimensional integrals in the objective function are replaced with tractable (generalized） moment problems [18,22,51]. In contrast, distributionally robust models with Wasserstein ambiguity sets are believed to be harder than their stochastic counterparts [36]. Indeed, the state-of-the-art method for computing the worst-case expectation over a Wasserstein ambiguity set $\mathcal { P }$ relies on global optimization techniques. Exploiting the fact that the extreme points of $\mathcal { P }$ are discrete distributions with a fixed number of atoms [52], one may reformulate the original worst-case expectation problem as a finite-dimensional non-convex program, which can be solved via “difference of convex programming” methods, see [52] or [36, Section 7.1].However, the computational effort is reported to be considerable, and there is no guarantee to find the global optimum. Nevertheless, tractability results are available for special cases. Specifically, the worst case of a convex law-invariant risk measure with respect to a Wasserstein ambiguity set $\mathcal { P }$ reduces to the sum of the nominal risk and a regularization term whenever $h ( x , \xi )$ is affine in $\xi$ and $\mathcal { P }$ does not include any support constraints [53]. Moreover, while this paper was under review we became aware of the PhD thesis [54], which reformulates a distributionally robust two-stage unit commitment problem over a Wasserstein ambiguity set as a semi-infinite linear program, which is subsequently solved using a Benders decomposition algorithm.

The main contribution of this paper is to demonstrate that the worst-case expectation over a Wasserstein ambiguity set can in fact be computed efficiently via convex optimization techniques for numerous loss functions of practical interest. Furthermore, we propose an efficient procedure for constructing an extremal distribution that attains the worst-case expectation-provided that such a distribution exists. Otherwise,we construct a sequence of distributions that attain the worst-case expectation asymptotically. As a by-product, our analysis shows that many interesting distributionally robust optimization problems with Wasserstein ambiguity sets can be solved in polynomial time. We also investigate the out-of-sample performance of the resulting optimal decisionsboth theoretically and experimentally—and analyze its dependence on the number of training samples. We highlight the following main contributions of this paper.

We prove that the worst-case expectation of an uncertain loss $\ell ( \xi )$ over a Wasserstein ambiguity set coincides with the optimal value of a finite-dimensional convex   
program if $\ell ( \xi )$ constitutes a pointwise maximum of finitely many concave functions.Generalizations to convex functions or to sums of maxima of concave functions are also discussed. We conclude that worst-case expectations can be computed efciently to high precision via modern convex optimization algorithms.   
We describe a supplementary finite-dimensional convex program whose optimal (near-optimal) solutions can be used to construct exact (approximate) extremal distributions for the infinite-dimensional worst-case expectation problem.   
We show that the worst-case expectation reduces to the optimal value of an explicit   
linear program if the 1-norm or the $\infty$ -norm is used in the definition of the Wasserstein metric and if $\ell ( \xi )$ belongs to any of the following function classes:（1） a   
pointwise maximum or minimum of affine functions; (2) the indicator function of   
a closed polytope or the indicator function of the complement of an open polytope; (3) the optimal value of a parametric linear program whose cost or right-hand side   
coefficients depend linearly on $\xi$   
Using recent measure concentration results from statistics, we demonstrate that the   
optimal value of a distributionally robust optimization problem over a Wasserstein ambiguity set provides an upper confidence bound on the out-of-sample cost of the   
worst-case optimal decision. We validate this theoretical performance guarantee in numerical tests.

If the uncertain parameter vector $\xi$ is confined to a fixed finite subset of $\mathbb { R } ^ { m }$ , then the worst-case expectation problems over Wasserstein ambiguity sets simplify substantially and can often be reformulated as tractable conic programs by leveraging ideas from robust optimization. An elegant second-order conic reformulation has been discovered, for instance, in the context of distributionally robust regression analysis [32], and a comprehensive list of tractable reformulations of distributionally robust risk constraints for various risk measures is provided in [39]. Our paper extends these tractability results to the practically relevant case where $\xi$ has uncountably many possible realizations-without resorting to space tessellation or discretization techniques that are prone to the curse of dimensionality.

When $\ell ( \xi )$ is linear and the distribution of $\xi$ ranges over a Wasserstein ambiguity set without support constraints, one can derive a concise closed-form expression for the worst-case risk of $\ell ( \xi )$ for various convex risk measures [53].However, these analytical solutions come at the expense of a loss of generality. We believe that the results of this paper may pave the way towards an efcient computational procedure for evaluating the worst-case risk of $\ell ( \xi )$ in more general setings where the loss function may be non-linear and $\xi$ may be subject to support constraints.

Among all metric-based ambiguity sets studied to date, the Kullback-Leibler ambiguity set has attracted most attention from the robust optimization community. It has first been used in financial portfolio optimization to capture the distributional uncertainty of asset returns with a Gaussian nominal distribution [19]. Subsequent work has focused on Kullback-Leibler ambiguity sets for discrete distributions with a fixed support, which offer additional modeling flexibility without sacrificing computational tractability [2,14]. It is also known that distributionally robust chance constraints involving a generic Kullback-Leibler ambiguity set are equivalent to the respective classical chance constraints under the nominal distribution but with a rescaled violation probability [26,27].Moreover, closed-form counterparts of distributionally robust expectation constraints with Kullback-Leibler ambiguity sets have been derived in [25].

However, Kullback-Leibler ambiguity sets typically fail to represent confidence sets for the unknown distribution $\mathbb { P }$ . To see this,assume that $\mathbb { P }$ is absolutely continuous with respect to the Lebesgue measure and that the ambiguity set is centered at the discrete empirical distribution ${ \widehat { \mathbb { P } } } _ { N }$ . Then,any distribution in a Kullback-Leibler ambiguity set around $\widehat { \mathbb { P } } _ { N }$ must assign positive probability mass to each training sample. As $\mathbb { P }$ has a density function, it must therefore reside outside of the Kullback-Leibler ambiguity set irrespective of the training samples. Thus, Kullback-Leibler ambiguity sets around ${ \widehat { \mathbb { P } } } _ { N }$ contain $\mathbb { P }$ with probability O. In contrast, Wasserstein ambiguity sets centered at ${ \widehat { \mathbb { P } } } _ { N }$ contain discrete as well as continuous distributions and, if properly calibrated, represent meaningful confidence sets for IP. We will exploit this property in Sect. 3 to derive finite-sample guarantees. A comparison and critical assessment of various metric-based ambiguity sets is provided in [45]. Specifically, it is shown that worst-case expectations over Kullback-Leibler and other divergence-based ambiguity sets are law invariant. In contrast, worst-case expectations over Wasserstein ambiguity sets are not. The law invariance can be exploited to evaluate worst-case expectations via the sample average approximation.

The models proposed in this paper fall within the scope of data-driven distributionally robust optimization [7,16,20,23]. Closest in spirit to our work is the robust sample average approximation [7], which seeks decisions that are robust with respect to the ambiguity set of all distributions that pass a prescribed statistical hypothesis test. Indeed, the distributions within the Wasserstein ambiguity set could be viewed as those that pass a multivariate goodness-of-fit test in light of the available training samples. This amounts to interpreting the Wasserstein distance between the empirical distribution ${ \widehat { \mathbb { P } } } _ { N }$ and a given hypothesis $\mathbb { Q }$ as a test statistic and the radius of the Wasserstein ambiguity set as a threshold that needs to be chosen in view of the test's desired significance level $\beta$ . The Wasserstein distance has already been used in tests for normality [17] and to devise nonparametric homogeneity tests [40].

The rest of the paper proceeds as follows. Section 2 sketches a generic framework for data-driven distributionally robust optimization, while Sect. 3 introduces our specific approach based on Wasserstein ambiguity sets and establishes its out-of-sample performance guarantees.In Sect. 4 we demonstrate that many worst-case expectation problems over Wasserstein ambiguity sets can be reduced to finite-dimensional convex programs,and we develop a systematic procedure for constructing worst-case distributions. Explicit linear programming reformulations of distributionally robust single and two-stage stochastic programs as well as uncertainty quantification problems are derived in Sect. 5. Section 6 extends the scope of the basic approach to broader classes of objective functions, and Sect. 7 reports on numerical results.

Notation We denote by $\mathbb { R } _ { + }$ the non-negative and by $\overline { { \mathbb { R } } } : = \mathbb { R } \cup \{ - \infty , \infty \}$ the extended reals. Throughout this paper, we adopt the conventions of extended arithmetics, whereby $\infty \cdot 0 = 0 \cdot \infty = 0 / 0 = 0$ and $\infty - \infty = - \infty + \infty = 1 / 0 = \infty$ The inner product of two vectors $a , b \in \mathbb { R } ^ { m }$ is denoted by $\langle a , b \rangle { : = } a ^ { \mathsf { T } } b$ . Given a norm $\| \cdot \|$ on $\mathbb { R } ^ { m }$ , the dual norm is defined through $\begin{array} { r } { \| z \| _ { * } { : = \operatorname* { s u p } _ { \| \xi \| \le 1 } \left. z , \xi \right. } } \end{array}$ . A function $f : \mathbb { R } ^ { m }  \overline { { \mathbb { R } } }$ is proper if $f ( \xi ) < + \infty$ for at least one $\xi$ and $f ( \xi ) > - \infty$ for every $\xi$ in $\mathbb { R } ^ { m }$ . The conjugate of $f$ is defined as $\begin{array} { r } { f ^ { * } ( z ) { : = \operatorname* { s u p } } _ { \xi \in \mathbb { R } ^ { m } } \left. z , \xi \right. - f ( \xi ) . } \end{array}$ Note that conjugacy preserves properness. For a set $\Xi \subseteq \mathbb { R } ^ { m }$ , the indicator function $\mathbb { 1 } _ { \Xi }$ is defined through $\mathbb { 1 } _ { \Xi } ( \xi ) = 1$ if $\xi \in \Xi ; = 0$ otherwise. Similarly, the characteristic function $\chi _ { \Xi }$ is defined via $\chi _ { \Xi } ( \xi ) = 0$ if $\xi \in \Xi ; = \infty$ otherwise. The support function of $\Xi$ is defined as $\sigma _ { \Xi } ( z ) { : = } \operatorname* { s u p } _ { \xi \in \Xi } \left. z , \xi \right.$ . It coincides with the conjugate of $\chi _ { \Xi }$ . We denote by $\delta _ { \xi }$ the Dirac distribution concentrating unit mass at $\boldsymbol { \xi } \in \mathbb { R } ^ { m }$ . The product of two probability distributions $\mathbb { P } _ { 1 }$ and $\mathbb { P } _ { 2 }$ on $\Xi _ { 1 }$ and $\Xi _ { 2 }$ , respectively, is the distribution $\mathbb { P } _ { 1 } \otimes \mathbb { P } _ { 2 }$ on $\Xi _ { 1 } \times \Xi _ { 2 }$ .The $N$ -fold product of a distribution $\mathbb { P }$ on $\Xi$ is denoted by $\mathbb { P } ^ { N }$ ,which represents a distribution on the Cartesian product space $\Xi ^ { N }$ . Finally, we set the expectation of $\ell : \Xi \to \overline { { \mathbb { R } } }$ under $\mathbb { P }$ t0 $\mathbb { E } ^ { \mathbb { P } } [ \ell ( \xi ) ] = \mathbb { E } ^ { \mathbb { P } } \big [ \operatorname* { m a x } \{ \ell ( \xi ) , 0 \} \big ] + \mathbb { E } ^ { \mathbb { P } } \big [ \operatorname* { m i n } \{ \ell ( \xi ) , 0 \} \big ]$ which is well-defined by the conventions of extended arithmetics.

# 2 Data-driven stochastic programming

Consider the stochastic program

$$
J ^ { \star } { : = } \operatorname* { i n f } _ { x \in \mathbb { X } } \left\{ \mathbb { E } ^ { \mathbb { P } } \big [ h ( x , \xi ) \big ] = \int _ { \Xi } h ( x , \xi ) \mathbb { P } ( \mathrm { d } \xi ) \right\}
$$

with feasible set $\mathbb { X } \subseteq \mathbb { R } ^ { n }$ , uncertainty set $\Xi \subseteq \mathbb { R } ^ { m }$ and loss function $h : \mathbb { R } ^ { n } \times \mathbb { R } ^ { m }  \overline { { \mathbb { R } } }$ The loss function depends both on the decision vector $x \in \mathbb { R } ^ { n }$ and the random vector $\xi \in \mathbb { R } ^ { m }$ ， whose distribution $\mathbb { P }$ is supported on E. Problem(1) can be viewed as the first-stage problem of a two-stage stochastic program, where $h ( x , \xi )$ represents the optimal value of a subordinate second-stage problem [46]. Alternatively, problem (1) may also be interpreted as a generic learning problem in the spirit of [49].

Unfortunately,in most situations of practical interest, the distribution $\mathbb { P }$ is not precisely known, and therefore we miss essential information to solve problem(1) exactly. However, $\mathbb { P }$ is often partially observable through a finite set of $N$ independent samples, e.g., past realizations of the random vector $\xi$ . We denote the training dataset comprising these samples by $\widehat { \Xi } _ { N } : = \{ \widehat { \xi } _ { i } \} _ { i \leq N } \subseteq \Xi$ .We emphasize that-before its revelation—the dataset $\widehat { \Xi } _ { N }$ can be viewed as a random object governed by the distribution $\mathbb { P } ^ { N }$ supported on $\Xi ^ { N }$ .

A data-driven solution for problem (1) is a feasible decision $\widehat { x } _ { N } \in \mathbb { X }$ that is constructed from the training dataset ${ \widehat { \Xi } } _ { N }$ . Throughout this paper, we notationally suppress the dependence of $\widehat { x } _ { N }$ on the training samples in order to avoid clutter. Instead, we reserve the superscript ‘\~’ for objects that depend on the training data and thus constitute random objects governed by the product distribution $\mathbb { P } ^ { N }$ . The out-of-sample performance of $\widehat { x } _ { N }$ is defined as $\mathbb { E } ^ { \mathbb { P } } \big [ h ( \widehat { x } _ { N } , \xi ) \big ]$ and can thus be viewed as the expected cost of $\widehat { x } _ { N }$ under a new sample $\xi$ that is independent of the training dataset. As $\mathbb { P }$ is unknown, however, the exact out-of-sample performance cannot be evaluated in practice,and the best we can hope for is to establish performance guarantees in the form of tight bounds. The feasibility of $\widehat { x } _ { N }$ in (1） implies $J ^ { \star } \le \mathrm { E } ^ { \mathrm { P } } \big [ h ( \widehat { x } _ { N } , \xi ) \big ]$ ,but this lower bound is again of limited use as $J ^ { \star }$ is unknown and as our primary concern is to bound the costs from above. Thus, we seek data-driven solutions $\widehat { x } _ { N }$ with performance guarantees of the type

$$
\begin{array} { r } { \mathbb { P } ^ { N } \Big \{ \widehat { \Xi } _ { N } : \mathbb { E } ^ { \mathbb { P } } \big [ h ( \widehat { x } _ { N } , \xi ) \big ] \leq \widehat { J } _ { N } \Big \} \geq 1 - \beta , } \end{array}
$$

where $\widehat { J _ { N } }$ constitutes an upper bound that may depend on the training dataset, and $\beta \in ( 0 , 1 )$ is a significance parameter with respect to the distribution $\mathbb { P } ^ { N }$ ，which governs both $\widehat { x } _ { N }$ and ${ \widehat { J } } _ { N }$ . Hereafter we refer to ${ \widehat { J } } _ { N }$ as a certificate for the out-ofsample performance of $\widehat { x } _ { N }$ and to the probability on the left-hand side of (2) as its reliability. Our ideal goal is to find a data-driven solution with the lowest possible out-of-sample performance.This is impossible,however, as $\mathbb { P }$ is unknown,and the out-of-sample performance cannot be computed. We thus pursue the more modest but achievable goal to find a data-driven solution with a low certificate and a high reliability.

A natural approach to generate data-driven solutions $\widehat { x } _ { N }$ is to approximate $\mathbb { P }$ with the discrete empirical probability distribution

$$
\widehat { \mathbb { P } } _ { N } { : = } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { \widehat { \xi } _ { i } } ,
$$

that is,the uniform distribution on $\widehat { \Xi } _ { N }$ . This amounts to approximating the original stochastic program (1) with the sample-average approximation (SAA) problem

$$
\widehat { J } _ { \mathrm { S A A } } : = \operatorname* { i n f } _ { x \in \mathbb { X } } \left\{ \mathbb { E } ^ { \widehat { \mathbb { P } } _ { N } } \big [ h ( x , \xi ) \big ] = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h ( x , \widehat { \xi } _ { i } ) \right\} .
$$

If the feasible set $\mathbb { X }$ is compact and the loss function is uniformly continuous in $x$ across all $\xi \in \Xi$ , then the optimal value and optimal solutions of the SAA problem (4) converge almost surely to their counterparts of the true problem(1） as $N$ tends to infinity [46, Theorem 5.3]. Even though finite sample performance guarantees of the type (2) can be obtained under additional assumptions such as Lipschitz continuity of the loss function (see e.g., [47, Theorem 1]), the SAA problem has been conceived primarily for situations where the distribution $\mathbb { P }$ is known and additional samples can be acquired cheaply via random number generation. However, the optimal solutions of the SAA problem tend to display a poor out-of-sample performance in situations where $N$ is small and where the acquisition of additional samples would be costly.

In this paper we address problem (1） with an alternative approach that explicitly accounts for our ignorance of the true data-generating distribution $\mathbb { P }$ , and that offers attractive performance guarantees even when the acquisition of additional samples from $\mathbb { P }$ is impossible or expensive. Specifically, we use ${ \widehat { \Xi } } _ { N }$ to design an ambiguity set ${ \widehat { \mathcal { P } } } _ { N }$ containing all distributions that could have generated the training samples with high confidence. This ambiguity set enables us to define the certificate $\widehat { J _ { N } }$ as the optimal value of a distributionally robust optimization problem that minimize the worst-case expected cost.

$$
\widehat { J } _ { N } { : = } \operatorname* { i n f } _ { x \in \mathbb { X } } \operatorname* { s u p } _ { \mathbb { Q } \in \widehat { \mathcal { P } } _ { N } } \mathbb { E } ^ { \mathbb { Q } } \big [ h ( x , \xi ) \big ]
$$

Following [38], we construct ${ \widehat { \mathcal { P } } } _ { N }$ as a ball around the empirical distribution (3） with respect to the Wasserstein metric.In the remainder of the paper we will demonstrate that the optimal value $\widehat { J } _ { N }$ as well as any optimal solution $\widehat { x } _ { N }$ (if it exists） of the distributionally robust problem (5） satisfy the following conditions.

(i) Finite sample guarantee: For a carefully chosen size of the ambiguity set, the certificate $\widehat { J _ { N } }$ provides a $1 - \beta$ confidence bound of the type (2) on the out-ofsample performance of $\widehat { x } _ { N }$   
(ii) Asymptotic consistency: As $N$ tends to infinity, the certificate $\widehat { J _ { N } }$ and the datadriven solution $\widehat { x } _ { N }$ converge—in a sense to be made precise below—to the optimal value $J ^ { \star }$ and an optimizer $x ^ { \star }$ of the stochastic program (1), respectively.   
(iii) Tractability: For many loss functions $h ( x , \xi )$ and sets $\mathbb { X }$ ，the distributionally robust problem (5) is computationally tractable and admits a reformulation reminiscent of the SAA problem (4).

Conditions (i-ii) have been identified in [7] as desirable properties of data-driven solutions for stochastic programs.Precise statements of these conditions will be provided in the remainder. In Sect. 3 we will use the Wasserstein metric to construct ambiguity sets of the type ${ \widehat { \mathcal { P } } } _ { N }$ satisfying the conditions (i) and (ii). In Sect. 4, we will demonstrate that these ambiguity sets also fulfill the tractability condition (ii). We see this last result as the main contribution of this paper because the state-of-the-art method for solving distributionally robust problems over Wasserstein ambiguity sets relies on global optimization algorithms [36].

# 3 Wasserstein metric and measure concentration

Probability metrics represent distance functions on the space of probability distributions. One of the most widely used examples is the Wasserstein metric, which is defined on the space $\mathcal { M } ( \Xi )$ of all probability distributions $\mathbb { Q }$ supported on $\Xi$ with $\begin{array} { r } { \mathbb { E } ^ { \mathbb { Q } } \big [ \| \xi \| \big ] = \int _ { \Xi } \bar { \| \xi \| } \mathbb { Q } ( \mathrm { d } \xi ) < \infty . } \end{array}$

Definition 3.1 (Wasserstein metric [29]) The Wasserstein metric $d _ { \mathrm { W } } : \mathcal { M } ( \Xi ) \times$ $\mathcal { M } ( \Xi )  \mathbb { R } _ { + }$ is defined via

for all distributions $\mathbb { Q } _ { 1 } , \mathbb { Q } _ { 2 } \in \mathcal { M } ( \Xi ) , w h e r e \| \cdot \|$ represents an arbitrary norm on $\mathbb { R } ^ { m }$

The decision variable $\Pi$ can be viewed as a transportation plan for moving a mass distribution described by $\mathbb { Q } _ { 1 }$ to another one described by $\mathbb { Q } _ { 2 }$ . Thus, the Wasserstein distance between $\mathbb { Q } _ { 1 }$ and $\mathbb { Q } _ { 2 }$ represents the cost of an optimal mass transportation plan, where the norm $\| \cdot \|$ encodes the transportation costs. We remark that a generalized $p$ Wasserstein metric for $p \geq 1$ is obtained by setting the transportation cost between $\xi _ { 1 }$ and $\xi _ { 2 }$ to $\| \xi _ { 1 } - \xi _ { 2 } \| ^ { p }$ . In this paper, however, we focus exclusively on the 1-Wasserstein metric of Definition 3.1, which is sometimes also referred to as the Kantorovich metric.

We will sometimes also need the following dual representation of the Wasserstein metric.

Theorem 3.2 (Kantorovich-Rubinstein [29]) For any distributions $\mathbb { Q } _ { 1 } , \mathbb { Q } _ { 2 } \in \mathcal { M } ( \Xi )$ we have

$$
d _ { \mathrm { W } } \big ( \mathbb { Q } _ { 1 } , \mathbb { Q } _ { 2 } \big ) = \operatorname* { s u p } _ { f \in \mathcal { L } } \Big \{ \int _ { \Xi } f ( \xi ) \mathbb { Q } _ { 1 } ( \mathrm { d } \xi ) - \int _ { \Xi } f ( \xi ) \mathbb { Q } _ { 2 } ( \mathrm { d } \xi ) \Big \} ,
$$

where $\mathcal { L }$ denotes the space of all Lipschitz functions with $| f ( \xi ) - f ( \xi ^ { \prime } ) | \le \| \xi - \xi ^ { \prime } \|$ for all $\xi , \xi ^ { \prime } \in \Xi$

Kantorovich and Rubinstein [29] originally established this result for distributions with bounded support. A modern proof for unbounded distributions is due to Villani [50, Remark 6.5,p. 107]. The optimization problems in Definition 3.1 and Theorem 3.2, which provide two equivalent characterizations of the Wasserstein metric, constitute a primal-dual pair of infinite-dimensional linear programs. The dual representation implies that two distributions $\mathbb { Q } _ { 1 }$ and $\mathbb { Q } _ { 2 }$ are close to each other with respect to the Wasserstein metric if and only if all functions with uniformly bounded slopes have similar integrals under $\mathbb { Q } _ { 1 }$ and $\mathbb { Q } _ { 2 }$ . Theorem 3.2 also demonstrates that the Wasserstein metric is a special instance of an integral probability metric (see e.g. [33]) and that its generating function class coincides with a family of Lipschitz continuous functions.

In the remainder we will examine the ambiguity set

$$
\begin{array} { r } { \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) { : = } \left\{ \mathbb { Q } \in \mathcal { M } ( \Xi ) \ : \ d _ { \mathbb { W } } \big ( \widehat { \mathbb { P } } _ { N } , \mathbb { Q } \big ) \leq \varepsilon \right\} , } \end{array}
$$

which can be viewed as the Wasserstein ball of radius $\varepsilon$ centered at the empirical distribution $\widehat { \mathbb { P } } _ { N }$ . Under a common light tail assumption on the unknown data-generating distribution $\mathbb { P }$ , this ambiguity set offers attractive performance guarantees in the spirit of Sect. 2.

Assumption 3.3 (Light-tailed distribution) There exists an exponent $a > 1$ such that

$$
A { : = } \mathbb { E } ^ { \mathbb { P } } \big [ \exp ( \| \xi \| ^ { a } ) \big ] = \int _ { \Xi } \exp ( \| \xi \| ^ { a } ) \mathbb { P } ( \mathrm { d } \xi ) < \infty .
$$

Assumption 3.3 essentially requires the tail of the distribution $\mathbb { P }$ to decay at an exponential rate.Note that this assumption trivially holds if $\Xi$ is compact. Heavytailed distributions that fail to meet Assumption 3.3 are difficult to handle even in the context of the classical sample average approximation. Indeed, under a heavy-tailed distribution the sample average of the loss corresponding to any fixed decision $x \in \mathbb { X }$ may not even converge to the expected loss; see e.g. [13,15]. The following modern measure concentration result provides the basis for establishing powerful finite sample guarantees.

Theorem 3.4 (Measure concentration [21, Theorem 2]) If Assumption 3.3 holds, we have

$$
\begin{array} { r } { \mathbb { P } ^ { N } \Big \{ d _ { \mathbb { W } } \big ( \mathbb { P } , \widehat { \mathbb { P } } _ { N } \big ) \geq \varepsilon \Big \} \leq \Big \{ \ L _ { c _ { 1 } } \exp \big ( { - c _ { 2 } N \varepsilon ^ { \mathrm { m a x } \{ m , 2 \} } } \big ) \quad i f \varepsilon \leq 1 , } \\ { c _ { 1 } \exp \big ( { - c _ { 2 } N \varepsilon ^ { a } } \big ) \quad \quad \quad \quad \quad \quad i f \varepsilon > 1 , } \end{array}
$$

for all $N \geq 1$ ， $m \neq 2$ ,and $\varepsilon > 0$ ,where $c _ { 1 }$ ， $c _ { 2 }$ are positive constants that only depend on a, $A$ ,and $m$ .1

Theorem 3.4 provides an a priori estimate of the probability that the unknown data-generating distribution $\mathbb { P }$ resides outside of the Wasserstein ball $\mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ . Thus, we can use Theorem 3.4 to estimate the radius of the smallest Wasserstein ball that contains $\mathbb { P }$ with confidence $1 - \beta$ for some prescribed $\beta \in ( 0 , 1 )$ . Indeed, equating the right-hand side of (7) to $\beta$ and solving for $\varepsilon$ yields

$$
\varepsilon _ { N } ( \beta ) { : = \left\{ \begin{array} { l l } { \left( \frac { \log ( c _ { 1 } \beta ^ { - 1 } ) } { c _ { 2 } N } \right) ^ { 1 / \operatorname* { m a x } \{ m , 2 \} } } & { \mathrm { i f ~ } N \geq \frac { \log ( c _ { 1 } \beta ^ { - 1 } ) } { c _ { 2 } } , } \\ { \left( \frac { \log ( c _ { 1 } \beta ^ { - 1 } ) } { c _ { 2 } N } \right) ^ { 1 / a } } & { \mathrm { i f ~ } N < \frac { \log ( c _ { 1 } \beta ^ { - 1 } ) } { c _ { 2 } } . } \end{array} \right. }
$$

Note that the Wasserstein ball with radius $\varepsilon _ { N } ( \beta )$ can thus be viewed as a confidence set for the unknown true distribution as in statistical testing; see also [7].

Theorem 3.5 (Finite sample guarantee) Suppose that Assumption 3.3 holds and that $\beta \in ( 0 , 1 )$ . Assume also that $\succnapprox$ and $\widehat { x } _ { N }$ represent the optimal value and an optimizer ofthe distributionally robust program (5) with ambiguity set $\widehat { \mathcal { P } } _ { N } = \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } )$ .Then, the finite sample guarantee (2) holds.

Proof The claim follows immediately from Theorem 3.4, which ensures via the definition of $\varepsilon _ { N } ( \beta )$ in (8) that $\mathbb { P } ^ { N } \{ \mathbb { P } \in \mathbf { \dot { \mathbb { B } } } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } ) \} \geq 1 - \beta$ Thus, $\mathbb { E } ^ { \mathbb { P } } [ h ( \widehat { x } _ { N } , \xi ) ] \leq$ $\begin{array} { r } { \operatorname* { s u p } _ { \mathbb { Q } \in \widehat { \mathcal { P } } _ { N } } \mathbb { E } ^ { \mathbb { Q } } [ h ( \widehat { x } _ { N } , \xi ) ] = \widehat { J } _ { N } } \end{array}$ with probability $1 - \beta$ □

It is clear from (8) that for any fixed $\beta > 0$ , the radius $\varepsilon _ { N } ( \beta )$ tends to O as $N$ increases. Moreover, one can show that if $\beta _ { N }$ converges to zero at a carefully chosen rate,then the solution of the distributionally robust optimization problem (5） with ambiguity set $\widehat { \mathcal { P } } _ { N } = \mathbb { B } _ { \varepsilon _ { N } ( \beta _ { N } ) } ( \widehat { \mathbb { P } } _ { N } )$ converges to the solution of the original stochastic program (1) as $N$ tends to infinity. The following theorem formalizes this statement.

Theorem 3.6 (Asymptotic consistency） Suppose that Assumption 3.3 holds and that $\beta _ { N } \in ( 0 , 1 )$ ， $N \in \mathbb { N } _ { : }$ satisfes $\textstyle \sum _ { N = 1 } ^ { \infty } \beta _ { N } < \infty$ and $\begin{array} { r } { \operatorname* { l i m } _ { N \to \infty } \varepsilon _ { N } ( \beta _ { N } ) = 0 } \end{array}$ 2 Assume also that $\widehat { J _ { N } }$ and $\widehat { x } _ { N }$ represent the optimal value and an optimizer ofthe distributionally robust program (5) with ambiguity set $\widehat { \mathcal { P } } _ { N } = \mathbb { B } _ { \varepsilon _ { N } ( \beta _ { N } ) } ( \widehat { \mathbb { P } } _ { N } ) , N \in \mathbb { N }$

(i）If $h ( x , \xi )$ is upper semicontinuous in $\xi$ and there exists $L \ge 0$ with $| h ( x , \xi ) | \overset { } { \underset { } { \le } }$ $L ( 1 + \| \xi \| )$ for all $x \in \mathbb { X }$ and $\xi \in \Xi$ ，then $\mathbb { P } ^ { \infty }$ -almost surely we have $\widehat { J _ { N } } \downarrow J ^ { \star }$ as $N \to \infty$ where $J ^ { \star }$ is the optimal value of (1).

(ii) If the assumptions of assertion $( i )$ hold, $\mathbb { X }$ is closed, and $h ( x , \xi )$ is lower semicontinuous in $x$ for every $\xi \in \Xi$ ， then any accumulation point of $\{ \widehat { x } _ { N } \} _ { N \in \mathbb { N } }$ is $\mathbb { P } ^ { \infty }$ -almost surely an optimal solution for (1).

The proof of Theorem 3.6 will rely on the following technical lemma.

Lemma 3.7 (Convergence of distributions) If Assumption 3.3 holds and $\beta _ { N } \in ( 0 , 1 )$ ， $N \in \mathbb { N }$ satisfies $\textstyle \sum _ { N = 1 } ^ { \infty } \beta _ { N } ~ < ~ \infty$ and $\mathrm { l i m } _ { N \to \infty } \varepsilon _ { N } ( \beta _ { N } ) = 0 $ then,anysequence $\widehat { \mathbb { Q } } _ { N } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta _ { N } ) } ( \widehat { \mathbb { P } } _ { N } ^ { - } ) ,$ $N \in \mathbb { N }$ where ${ \widehat { \mathbb { Q } } } _ { N }$ may depend on the training data,converges under the Wasserstein metric (and thus weakly) to $\mathbb { P }$ almost surely with respect to $\mathbb { P } ^ { \infty }$ ， that is,

$$
\mathbb { P } ^ { \infty } \left\{ \operatorname* { l i m } _ { N \to \infty } d _ { \mathbb { W } } \big ( \mathbb { P } , \widehat { \mathbb { Q } } _ { N } \big ) = 0 \right\} = 1 .
$$

Proof As $\widehat { \mathbb { Q } } _ { N } \in \mathbb { B } _ { \delta _ { N } } ( \widehat { \mathbb { P } } _ { N } )$ ,the triangle inequality for the Wasserstein metric ensures that

$$
\begin{array} { r } { d _ { \mathbb { W } } \big ( \mathbb { P } , \widehat { \mathbb { Q } } _ { N } \big ) \leq d _ { \mathbb { W } } \big ( \mathbb { P } , \widehat { \mathbb { P } } _ { N } \big ) + d _ { \mathbb { W } } \big ( \widehat { \mathbb { P } } _ { N } , \widehat { \mathbb { Q } } _ { N } \big ) \leq d _ { \mathbb { W } } \big ( \mathbb { P } , \widehat { \mathbb { P } } _ { N } \big ) + \varepsilon _ { N } ( \beta _ { N } ) . } \end{array}
$$

Moreover, Theorem 3.4 implies that $\mathbb { P } ^ { N } \{ d _ { \mathrm { W } } \big ( \mathbb { P } , \widehat { \mathbb { P } } _ { N } \big ) \le \varepsilon _ { N } ( \beta _ { N } ) \} \ge 1 - \beta _ { N }$ ,and thus we have $\mathbb { P } ^ { N } \{ d _ { \mathbb { W } } \big ( \mathbb { P } , \widehat { \mathbb { Q } } _ { N } \big ) \le 2 \varepsilon _ { N } ( \beta _ { N } ) \} \ge 1 - \beta _ { N }$ .As $\textstyle \sum _ { N = 1 } ^ { \infty } \beta _ { N } ~ < ~ \infty$ ,the Borel-Cantelli Lemma [28, Theorem 2.18] further implies that

$$
\begin{array} { r } { \mathbb { P } ^ { \infty } \left\{ d _ { \mathsf { W } } \big ( \mathbb { P } , \widehat { \mathbb { Q } } _ { N } \big ) \leq \varepsilon _ { N } ( \beta _ { N } ) \mathrm { ~ f o r ~ a l l ~ s u f f c i e n t l y ~ l a r g e ~ } N \right\} = 1 . } \end{array}
$$

Finally, as $\begin{array} { r } { \operatorname* { l i m } _ { N \to \infty } \varepsilon _ { N } ( \beta _ { N } ) = 0 } \end{array}$ , we conclude that $\mathrm { l i m } _ { N \to \infty } d _ { \mathrm { W } } \big ( \mathbb { P } , \widehat { \mathbb { Q } } _ { N } \big ) = 0$ almost surely. Note that convergence with respect to the Wasserstein metric implies weak convergence [10]. □

Proof of Theorem 3.6 As $\widehat { x } _ { N } \in \mathbb { X }$ ，we have $J ^ { \star } \le \mathbb { E } ^ { \mathbb { P } } [ h ( \widehat { x } _ { N } , \xi ) ]$ .Moreover, Theorem 3.5 implies that

$$
\begin{array} { r } { \mathbb { P } ^ { N } \left\{ J ^ { \star } \leq \mathbb { E } ^ { \mathbb { P } } [ h ( \widehat { x } _ { N } , \xi ) ] \leq \widehat { J } _ { N } \right\} \geq \mathbb { P } ^ { N } \left\{ \mathbb { P } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta _ { N } ) } ( \widehat { \mathbb { P } } _ { N } ) \right\} \geq 1 - \beta _ { N } , } \end{array}
$$

for all $N \in \mathbb { N }$ As $\textstyle \sum _ { N = 1 } ^ { \infty } \beta _ { N } < \infty$ ,the Borel-CantelliLemma further implies that

$$
\begin{array} { r } { \mathbb { P } ^ { \infty } \left\{ J ^ { \star } \le \mathbb { E } ^ { \mathbb { P } } [ h ( \widehat { x } _ { N } , \xi ) ] \le \widehat { J } _ { N } \mathrm { ~ f o r ~ a l l ~ s u f f c i e n t l y ~ l a r g e ~ } N \right\} = 1 . } \end{array}
$$

To prove assertion (i), it thus remains to be shown that lim $\mathrm { s u p } _ { N \to \infty } { \widehat { J } } _ { N } \leq J ^ { \star }$ with probability $1 . \operatorname { A s } h ( x , \xi )$ is upper semicontinuous and grows at most linearly in $\xi$ , there exists a non-increasing sequence of functions $h _ { k } ( x , \xi )$ ， $k \in \mathbb N$ ，such that $h ( x , \xi ) =$ $\scriptstyle \operatorname* { l i m } _ { k \to \infty } h _ { k } ( x , \xi )$ ,and $h _ { k } ( x , \xi )$ is Lipschitz continuous in $\xi$ for any fixed $x \in \mathbb { X }$ and $k \in \mathbb N$ with Lipschitz constant $L _ { k } \ge 0$ ; see Lemma A.1 in the appendix. Next, choose any $\delta > 0$ ,fix a $\delta$ -optimal decision $x _ { \delta } \in \mathbb { X }$ for (1) with $\mathbb { E } ^ { \mathbb { P } } [ h ( x _ { \delta } , \xi ) ] \le J ^ { \star } + \delta$ ,and for every $N \in \mathbb N$ let $\widehat { \mathbb { Q } } _ { N } \in \widehat { \mathcal { P } } _ { N }$ be a $\delta$ -optimal distribution corresponding to $x _ { \delta }$ with

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \widehat { \mathcal { P } } _ { N } } \mathbb { E } ^ { \mathbb { Q } } [ h ( x _ { \delta } , \xi ) ] \leq \mathbb { E } ^ { \mathbb { Q } _ { N } } [ h ( x _ { \delta } , \xi ) ] + \delta .
$$

Then, we have

$$
\begin{array} { r l } { \underset { N  \infty } { \operatorname* { l i m } } \underset { 0 \leq \infty } { \operatorname* { s u p } } \ f _ { N } \leq \underset { N  \infty } { \operatorname* { l i m } } \ \underset { 0 \leq \infty } { \operatorname* { s u p } } \ \underset { 0 \leq \infty } { \operatorname* { E q } } [ h ( x _ { \delta } , \xi ) ] } & { } \\ & { \leq \underset { N  \infty } { \operatorname* { l i m } } \ \underset { 0 \leq \infty } { \operatorname* { s u p } } \ \underset { 0 \leq \infty } { \operatorname* { f i r } } [ h ( x _ { \delta } , \xi ) ] + \delta } \\ & { \leq \underset { k  \infty } { \operatorname* { l i m } } \ \underset { N  \infty } { \operatorname* { l i m } } \ \underset { 0 \leq \infty } { \operatorname* { f i r } } [ h _ { k } ( x _ { \delta } , \xi ) ] + \delta } \\ & { \leq \underset { k  \infty } { \operatorname* { l i m } } \ \underset { N  \infty } { \operatorname* { l i m } } ( \mathrm { E } ^ { \mathrm { P } } [ h _ { k } ( x _ { \delta } , \xi ) ] + L _ { k } d _ { \mathrm { w } } ( \mathrm { P } , \widehat { \mathbb { Q } } _ { N } ) ) + \delta } \\ & { = \underset { k  \infty } { \operatorname* { l i m } } \ \underset { k  \infty } { \operatorname* { F } } [ h _ { k } ( x _ { \delta } , \xi ) ] + \delta , \mathbb { P } ^ { \infty } - \mathrm { a l m o s t } \ \mathrm { s u r e l y } } \\ & { \qquad \quad = \mathrm { E } ^ { \mathrm { P } } [ h ( x _ { \delta } , \xi ) ] + \delta \leq J ^ { \kappa } + 2 \delta , } \end{array}
$$

where the second inequality holds because $h _ { k } ( x , \xi )$ converges from above to $h ( x , \xi )$ ， and the third inequality follows from Theorem 3.2. Moreover, the almost sure equality holds due to Lemma 3.7,and the last equality follows from the Monotone Convergence Theorem [30, Theorem 5.5],which applies because $| \mathbb { E } ^ { \mathbb { P } } [ h _ { k } ( x _ { \delta } , \xi ) ] | < \infty$ Indeed, recall that $\mathbb { P }$ has an exponentially decaying tail due to Assumption 3.3 and that $h _ { k } ( x _ { \delta } , \xi )$ is Lipschitz continuous in $\xi$ . As $\delta > 0$ was chosen arbitrarily, we thus conclude that lim $\mathrm { { { \widehat { s u p } } } } _ { N  \infty } { \widehat { J } } _ { N } \leq J ^ { \star }$

To prove assertion (i),fix an arbitrary realization of the stochastic process $\{ { \widehat { \xi } } _ { N } \} _ { N \in { \mathbb { N } } }$ such that $J ^ { \star } = \operatorname* { l i m } _ { N \to \infty } \widehat { J } _ { N }$ and $J ^ { \star } \le \mathbb { E } ^ { \mathbb { P } } [ h ( \widehat { x } _ { N } , \xi ) ] \le \widehat { J } _ { N }$ for all sufficiently large $N$ . From the proof of assertion (i) we know that these two conditions are satisfied $\mathbb { P } ^ { \infty }$ -almost surely. Using these assumptions,one easily verifies that

$$
\operatorname* { l i m } _ { N  \infty } \operatorname* { i n f } _ { \mathbb { E } } { \mathbb { E } } ^ { \mathbb { P } } [ h ( \widehat { x } _ { N } , \xi ) ] \leq \operatorname* { l i m } _ { N  \infty } \widehat { J } _ { N } = J ^ { \star } .
$$

Next, let $x ^ { \star }$ be an accumulation point of the sequence $\{ \widehat { x } _ { N } \} _ { N \in \mathbb { N } }$ ,and note that $x ^ { \star } \in \mathbb { X }$ as $\mathbb { X }$ is closed. By passing to a subsequence, if necessary, we may assume without loss of generality that $x ^ { \star } = \operatorname* { l i m } _ { N \to \infty } \widehat { x } _ { N }$ .Thus,

$$
J ^ { \star } \le \mathbb { E } ^ { \mathbb { P } } [ h ( x ^ { \star } , \xi ) ] \le \mathbb { E } ^ { \mathbb { P } } [ \operatorname* { l i m } _ { N \to \infty } h ( \widehat { x } _ { N } , \xi ) ] \le \operatorname* { l i m } _ { N \to \infty } \mathbb { E } ^ { \mathbb { P } } [ h ( \widehat { x } _ { N } , \xi ) ] \le J ^ { \star } ,
$$

where the first inequality exploits that $x ^ { \star } \in \mathbb { X }$ , the second inequality follows from the lower semicontinuity of $h ( x , \xi )$ in $x$ , the third inequality holds due to Fatou's lemma (which applies because $h ( x , \xi )$ grows at most linearly in $\xi$ ),and the last inequality follows from (9). Therefore, we have $\mathbb { E } ^ { \mathbb { P } } [ h ( x ^ { \star } , \xi ) ] = J ^ { \star }$ □

In the following we show that all assumptions of Theorem 3.6 are necessary for asymptotic convergence, that is,relaxing any of these conditions can invalidate the convergence result.

Example $I$ (Necessity of regularity conditions)

(1） Upper semicontinuity of $\xi \mapsto h ( x , \xi )$ in Theorem $3 . 6 ( i )$ ：

Set $\Xi = [ 0 , 1 ] , \mathbb { P } = \delta _ { 0 }$ and $h ( x , \xi ) \lrcorner = \textbf { 1 } _ { ( 0 , 1 ] } ( \xi )$ ，whereby $J ^ { \star } = 0$ As $\mathbb { P }$ concentrates unit mass at O, we have $\widehat { \mathbb { P } } _ { N } = \delta _ { 0 } = \mathbb { P }$ irrespective of $N \in \mathbb { N }$ For any $\varepsilon > 0$ , the Dirac distribution $\delta _ { \varepsilon }$ thus resides within the Wasserstein ball $\mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ . Hence, $\widehat { J _ { N } }$ fails to converge to $J ^ { \star }$ for $\varepsilon \to 0$ because

$$
\begin{array} { r } { \widehat { J } _ { N } \geq \mathbb { E } ^ { \delta _ { \varepsilon } } [ h ( x , \xi ) ] = h ( x , \varepsilon ) = 1 , \forall \varepsilon > 0 . } \end{array}
$$

(2) Linear growth of $\xi \mapsto h ( x , \xi )$ in Theorem $3 . 6 ( i )$ ：

Set $\Xi = \mathbb { R }$ ， $\mathbb { P } = \delta _ { 0 }$ and $h ( x , \xi ) = \xi ^ { 2 }$ ,which implies that $J ^ { \star } = 0$ . Note that for any $\rho > \varepsilon$ , the two-point distribution $\begin{array} { r } { \mathbb { Q } _ { \rho } = ( 1 - \frac { \varepsilon } { \rho } ) \delta _ { 0 } + \frac { \varepsilon } { \rho } \delta _ { \rho } } \end{array}$ is contained in the Wasserstein ball $\mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ of radius $\varepsilon > 0$ . Hence, $\widehat { J _ { N } }$ fails to converge to $J ^ { \star }$ for $\varepsilon \to 0$ because

$$
{ \widehat { J _ { N } } } \geq \operatorname* { s u p } _ { \rho > \varepsilon } \mathbb { E } ^ { \mathbb { Q } _ { \rho } } [ h ( x , \xi ) ] = \operatorname* { s u p } _ { \rho > \varepsilon } \varepsilon \rho = \infty , \forall \varepsilon > 0 .
$$

(3) Lower semicontinuity of $x \mapsto h ( x , \xi )$ in Theorem 3.6 (ii):

Set $\mathbb { X } = [ 0 , 1 ]$ and $h ( x , \xi ) = \mathbb { 1 } _ { [ 0 . 5 , 1 ] } ( x )$ ，whereby $J ^ { \star } = 0$ irrespective of $\mathbb { P }$ As the objective is independent of $\xi$ ， the distributionally robust optimization problem (5) is equivalent to (1). Then, $\begin{array} { r } { \widehat { x } _ { N } = \frac { N - 1 } { 2 N } } \end{array}$ is a sequence of minimizers for (5) whose accumulation point $\begin{array} { r } { x ^ { \star } = \frac { 1 } { 2 } } \end{array}$ fails to be optimal in (1).

A convergence result akin to Theorem 3.6 for goodness-of-fit-based ambiguity sets is discussed in [7, Section 4]. This result is complementary to Theorem 3.6. Indeed, Theorem 3.6(i) requires $h ( x , \xi )$ to be upper semicontinuous in $\xi$ ，which is a necessary condition in our seting (see Example 1) that is absent in [7]. Moreover, Theorem 3.6(ii) only requires $h ( x , \xi )$ to be lower semicontinuous in $x$ , while [7] asks for equicontinuity of this mapping. This stronger requirement provides a stronger result, that is,the almost sure convergence of $\begin{array} { r } { \operatorname* { s u p } _ { \mathbb { Q } \in \widehat { \mathcal { P } } _ { N } } \mathbb { E } ^ { \mathbb { Q } } [ h ( x , \bar { \xi } ) ] } \end{array}$ t0 $\mathbb { E } ^ { \mathbb { P } _ { [ h ( x , \xi ) ] } }$ uniformly in $x$ on any compact subset of $\mathbb { X }$

Theorems 3.5 and 3.6 indicate that a careful a priori design of the Wasserstein ball results in attractive finite sample and asymptotic guarantees for the distributionally robust solutions. In practice, however, setting the Wasserstein radius to $\varepsilon _ { N } ( \beta )$ yields over-conservative solutions for the following reasons:

· Even though the constants $c _ { 1 }$ and $c _ { 2 }$ in (8) can be computed based on the proof of [21, Theorem 2], the resulting Wasserstein ball is larger than necessary, i.e., $\mathbb { P } \not \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } )$ with probability $\ll \beta$   
· Even if $\widetilde { \mathbb { P } } \not \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } )$ , the optimal value $\widehat { J _ { N } }$ of (5) may still provide an upper bound on $J ^ { \star }$ .   
· The formula for $\varepsilon _ { N } ( \beta )$ in (8) is independent of the training data. Allowing for random Wasserstein radii, however, results in a more efficient use of the available training data.

While Theorems 3.5 and 3.6 provide strong theoretical justification for using Wasserstein ambiguity sets, in practice, it is prudent to calibrate the Wasserstein radius via bootstrapping or cross-validation instead of using the conservative a priori bound $\varepsilon _ { N } ( \beta )$ ; see Sect. 7.2 for further details. A similar approach has been advocated in [7] to determine the sizes of ambiguity sets that are constructed via goodness-of-fit tests.

So far we have seen that the Wasserstein metric allows us to construct ambiguity sets with favorable asymptotic and finite sample guarantees. In the remainder of the paper we will further demonstrate that the distributionally robust optimization problem (5) with a Wasserstein ambiguity set (6) is not significantly harder to solve than the corresponding SAA problem (4).

# 4 Solving worst-case expectation problems

We now demonstrate that the inner worst-case expectation problem in (5） over the Wasserstein ambiguity set (6) can be reformulated as a finite convex program for many loss functions $h ( x , \xi )$ of practical interest.For ease of notation, throughout this section we suppress the dependence on the decision variable $x$ . Thus, we examine a generic worst-case expectation problem

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ]
$$

involving a decision-independent loss function $\ell ( \xi ) { : = } \operatorname* { m a x } _ { k \leq K } \ell _ { k } ( \xi )$ , which is defined as the pointwise maximum of more elementary measurable functions $\ell _ { k } : \mathbb { R } ^ { m } \to \overline { { \mathbb { R } } }$ ， $k \ \leq \ K$ . The focus on loss functions representable as pointwise maxima is nonrestrictive unless we impose some structure on the functions $\ell _ { k }$ . Many tractability results in the remainder of this paper are predicated on the following convexity assumption.

Assumption 4.1 (Convexity） The uncertainty set $\Xi \subseteq \mathbb { R } ^ { m }$ is convex and closed, and the negative constituent functions $- \ell _ { k }$ are proper, convex, and lower semicontinuous for all $k \leq K$ . Moreover, we assume that $\ell _ { k }$ is not identically $- \infty$ on $\Xi$ for all $\le K$ ·

Assumption 4.1 essentially stipulates that $\ell ( \xi )$ can be written as a maximum of concave functions. As we will showcase in Sect. 5, this mild restriction does not sacrifice much modeling power. Moreover, generalizations of this setting will be discussed in Sect. 6. We proceed as follows. Sect. 4.1 addresses the reduction of (10) to a finite convex program, while Sect. 4.2 describes a technique for constructing worst-case distributions.

# 4.1 Reduction to a finite convex program

The worst-case expectation problem (1O) constitutes an infinite-dimensional optimization problem over probability distributions and thus appears to be intractable.However, we will now demonstrate that (1O) can be re-expressed as a finite-dimensional convex program by leveraging tools from robust optimization.

Theorem 4.2 (Convex reduction) If the convexity Assumption 4.1 holds, then for any $\varepsilon \geq 0$ the worst-case expectation (1O) equals the optimal value of the finite convex program

$$
\left\{ \begin{array} { l l } { ~ \displaystyle \operatorname* { i n f } _ { \lambda , s _ { i } , z _ { i k } , \nu _ { i k } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \mathrm { s . t . } \qquad [ - \ell _ { k } ] ^ { * } ( z _ { i k } - \nu _ { i k } ) + \sigma _ { \Xi } ( \nu _ { i k } ) - \left. z _ { i k } , \widehat { \xi } _ { i } \right. \leq s _ { i } \forall i \leq N , \quad \forall k \leq K } \\ { \| z _ { i k } \| _ { * } \leq \lambda } & { \forall i \leq N , \quad \forall k \leq K . } \end{array} \right.
$$

Recall that $[ - \ell _ { k } ] ^ { * } ( z _ { i k } - \nu _ { i k } )$ denotes the conjugate of $- \ell _ { k }$ evaluated at $z _ { i k } - \nu _ { i k }$ and $\| z _ { i k } \| _ { * }$ the dual norm of $z _ { i k }$ . Moreover, $\chi _ { \Xi }$ represents the characteristic function of $\Xi$ and $\sigma _ { \Xi }$ its conjugate, that is, the support function of $\Xi$

Proof of Theorem 4.2 By using Definition 3.1 we can re-express the worst-case expectation (10) as

$$
\begin{array} { r l } { \underset { Q \in \mathbb { R } _ { \delta } ( \widehat { \mathbb { P } } _ { N } ) } { \operatorname* { s u p } } } & { \mathrm { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ] = \left\{ \begin{array} { l l } { \underset { \mathrm { 0 : t } , Q } { \operatorname* { s u p } } \int _ { \Xi } \ell ( \xi ) \operatorname { Q } ( \mathrm { d } \xi ) } \\ { \overset { \mathrm { s . t . } } { \operatorname { s . t } } \int _ { \Xi ^ { 2 } } \| \xi - \xi ^ { \prime } \| \operatorname { I n } ( \mathrm { d } \xi , \mathrm { d } \xi ^ { \prime } ) \leq \varepsilon } \\ { \quad } \\ { \quad } \\ { \quad } \end{array} \right. } \\ & { = \left\{ \begin{array} { l l } { \underset { \mathrm { 0 : t } + \mathtt { h } + \mathtt { h } + \mathtt { h } + \mathtt { h } + \mathtt { h } + \mathtt { h } \times \mathtt { m } + \mathtt { h } \times \mathtt { m } + \mathtt { h } \times \mathtt { m } + \xi ^ { \prime } } { \operatorname { s u p } } } \\ { \underset { \mathrm { 0 : t } \leq M ( \Xi ) } { \operatorname* { s u p } } \frac { 1 } { N } \int _ { \Xi } \ell ( \xi ) \operatorname { Q } _ { i } ( \mathrm { d } \xi ) } \\ { \quad } \\ { \mathrm { s . t . } \quad } \\ { \quad } \end{array} \right. } \end{array}
$$

The second equality follows from the law of total probability, which asserts that any joint probability distribution $\Pi$ of $\xi$ and $\xi ^ { \prime }$ can be constructed from the marginal distribution $\widehat { \mathbb { P } } _ { N } ^ { \phantom { \dagger } }$ of $\xi ^ { \prime }$ and the conditional distributions $\mathbb { Q } _ { i }$ of $\xi$ given $\xi ^ { \prime } = \widehat { \xi } _ { i } , i \leq N$ ， that is, we may wrte $\begin{array} { r } { \Pi = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { \widehat { \xi } _ { i } } \otimes \mathbb { Q } _ { i } } \end{array}$ Theresulinoe represents a generalized moment problem in the distributions $\mathbb { Q } _ { i }$ ， $i ~ \leq ~ N$ . Using a standard duality argument, we obtain

$$
\begin{array} { r l } { \underset { \mathbb { Q } \in \mathbb { S } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } { \operatorname* { s u p } } \mathbb { E } ^ { \mathbb { Q } } \big [ \varepsilon ( \xi ) \big ] = \underset { \mathbb { Q } _ { \varepsilon } \times \mathbb { M } ( \widehat { \mathbb { P } } ) } { \operatorname* { s u p } } \underset { i = 1 } { \operatorname* { i n f } } \frac { 1 } { N } \underset { i = 1 } { \overset { N } { \sum } } \int _ { \Xi } \ell ( \xi ) \mathbb { Q } _ { i } ( \mathbb { Q } \mathbb { E } ) } \\ & { \ + \lambda \bigg ( \varepsilon - \frac { 1 } { N } \underset { i = 1 } { \overset { N } { \sum } } \int _ { \Xi } \| \xi - \widehat { \mathbb { g } } _ { i } \| \mathbb { Q } _ { i } ( \mathbb { Q } \xi ) \bigg ) } \\ & { \ \leq \underset { \varepsilon \leq 0 } { \operatorname* { i n f } } \underset { \mathbb { Q } \in \mathcal { A } ( \mathbb { Q } ) } { \operatorname* { s u p } } \ \underset { i \leq N } { \operatorname* { i n f } } \big ( \lambda \underset { i = 1 } { \overset { N } { \sum } } \int _ { \Xi } \big ( \varepsilon ( \xi ) - \lambda \| \xi - \widehat { \mathbb { g } } _ { i } \| \big ) \ Q _ { i } ( \mathbb { Q } \xi ) } \\ & { \ \overset { ( 1 2 \lambda ) } { = } \underset { \varepsilon \leq 0 } { \operatorname* { i n f } } \lambda \varepsilon + \frac { 1 } { N } \underset { i = 1 } { \overset { N } { \sum } } \underset { \mathbb { S } \in \Xi } { \operatorname* { i n f } } \big ( \varepsilon ( \xi ) - \lambda \| \xi - \widehat { \mathbb { g } } _ { i } \| \big ) , \ \qquad ( 1 2 \lambda ) } \end{array}
$$

where(12a) follows from the max-min inequality,and (12b) follows from the fact that $\mathcal { M } ( \Xi )$ contains all the Dirac distributions supported on $\Xi$ . Introducing epigraphical auxiliary variables $s _ { i }$ ， $i \leq N$ , allows us to reformulate (12b) as

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { \lambda , s _ { i } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \displaystyle \operatorname { s . t . } \operatorname* { s u p } _ { \xi \in \Xi } \left( \ell ( \xi ) - \lambda \| \xi - \widehat { \xi } _ { i } \| \right) \leq s _ { i } \quad \forall i \leq N } \\ { \displaystyle \quad \lambda \geq 0 } \end{array} \right.
$$

$$
\begin{array} { r l } & { = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { \hat { x } \neq \hat { x } } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \displaystyle \operatorname* { s t } _ { \hat { x } \neq \hat { x } } \operatorname* { s u p } _ { i \in \hat { x } } \left( \varepsilon _ { k } ( \xi ) - \underset { | z _ { i } | \leq \hat { x } } { \operatorname* { m a x } } \left. z _ { i k } , \xi - \widehat { \xi } _ { i } \right. \right) \leq s _ { i } } & { \forall i \leq N , \forall k \leq K } \\ { \displaystyle \quad \quad \quad \quad \quad \lambda \geq 0 } \end{array} \right. } \\ & { \leq \left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { \hat { x } , \hat { x } \neq \hat { x } } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \displaystyle \operatorname* { s t } _ { \hat { x } \neq \hat { x } } \operatorname* { m i n } _ { \hat { x } \in \mathbb { S } } \left( \varepsilon _ { k } ( \xi ) - \left. z _ { i k } , \xi - \widehat { \xi } _ { i } \right. \right) \leq s _ { i } } & { \forall i \leq N , \forall k \leq K } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \lambda \geq 0 . } \end{array} \right. } \end{array}
$$

Equality (12d) exploits the definition of the dual norm and the decomposability of $\ell ( \xi )$ into its constituents $\ell _ { k } ( \xi ) , k \le K$ . Interchanging the maximization over $z _ { i k }$ with the minus sign (thereby converting the maximization to a minimization) and then with the maximization over $\xi$ leads to a restriction of the feasible set of (12d). The resulting upper bound (12e) can be re-expressed as

$$
\begin{array} { r l } & { \quad \quad \displaystyle \left. \begin{array} { l l l } { \displaystyle \operatorname* { i n f } _ { \lambda , s _ { i } , z _ { k } } } & { \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \mathit { s . s . } \varepsilon _ { i } } & { \operatorname* { s u p } } \end{array} \right. } \\ & { \quad \quad \quad \quad \quad \xi \in \Xi \ \left(  { \varepsilon } _ { k } ( \xi ) - \left. z _ { i k } , \xi \right. \right) + \left. z _ { i k } , \widehat { \xi } _ { i } \right. \leq s _ { i } \quad \forall i \leq N , \forall k \leq K } \\ & { \quad \quad \quad \| z _ { i k } \| _ { * } \leq \lambda } \\ & { = \left. \begin{array} { l l l } { \displaystyle \operatorname* { i n f } _ { \lambda , s _ { i } , z _ { k } } } & { \displaystyle \frac { N } { i + N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \displaystyle \operatorname* { s . t } _ { i } } & { \displaystyle - \ell _ { k } \leq k \sum ^ { 1 } ( z _ { i k } ) - \left. z _ { i k } , \widehat { \xi } _ { i } \right. \leq s _ { i } } & { \forall i \leq N , \forall k \leq K } \\ { \displaystyle \operatorname* { s t } _ { i } \leq N _ { * } \forall k \leq K , } & { \forall i \leq N , \forall k \leq K , } \end{array} \right. } \end{array}
$$

where(12f) follows from the definition of conjugacy, our conventions of extended arithmetic,and the substitution of $z _ { i k }$ with $- z i k$ . Note that (12f) is already a finite convex program.

Next,we show that Assumption 4.1 reduces the inequalities (12a) and(12e） to equalities. Under Assumption 4.1, the inequality (12a) is in fact an equality for any $\varepsilon \ > \ 0$ by virtue of an extended version of a well-known strong duality result for moment problems [44, Proposition 3.4]. One can show that (12a) continues to hold as an equality even for $\varepsilon = 0$ , in which case the Wasserstein ambiguity set (6) reduces to the singleton $\{ { \widehat { \mathbb { P } } } _ { N } \}$ , while (10)reduces tothesampleaverage $\overline { { { 1 } } } \ \sum _ { i = 1 } ^ { N } \widehat { \ell } ( \widehat { \xi _ { i } } )$ Indeed, for $\varepsilon \ = \ 0$ the variable $\lambda$ in (12b） can be increased indefinitely at no penalty. As $\ell ( \xi )$ constitutes a pointwise maximum of upper semicontinuous concave functions, $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( \widehat { \xi } _ { i } ) } \end{array}$ bu $\lambda$

The inequality (12e) also reduces to an equality under Assumption 4.1 thanks to the classical minimax theorem [4, Proposition 5.5.4], which applies because the set $\{ z _ { i k } \in \mathbb { R } ^ { m } : \| z _ { i k } \| _ { * } \leq \lambda \}$ is compact for any finite $\lambda \geq 0$ . Thus, the optimal values of (10) and (12f) coincide.

Assumption 4.1 further implies that the function $- \ell _ { k } + \chi _ { \Xi }$ is proper, convex and lower semicontinuous.Properness holds because $\ell _ { k }$ is not identically $- \infty$ on $\Xi$ By Rockafellar and Wets [42, Theorem 11.23(a), p. 493], its conjugate essentially coincides with the epi-addition (also known as inf-convolution) of the conjugates of the functions $- \ell _ { k }$ and $\sigma _ { \Xi }$ . Thus,

$$
\begin{array} { r l } & { [ - \ell _ { k } + \chi _ { \Xi } ] ^ { * } ( z _ { i k } ) = \operatorname* { i n f } _ { \nu _ { i k } } \Big ( [ - \ell _ { k } ] ^ { * } ( z _ { i k } - \nu _ { i k } ) + [ \chi _ { \Xi } ] ^ { * } ( \nu _ { i k } ) \Big ) } \\ & { \qquad = \mathrm { c l } \Big [ \operatorname* { i n f } _ { \nu _ { i k } } \Big ( [ - \ell _ { k } ] ^ { * } ( z _ { i k } - \nu _ { i k } ) + \sigma _ { \Xi } ( \nu _ { i k } ) \Big ) \Big ] , } \end{array}
$$

where cl[-] denotes the closure operator that maps any function to its largest lower semicontinuous minorant. As $\operatorname { c l } [ f ( \xi ) ] \leq 0$ if and only if $f ( \xi ) \leq 0$ for any function $f$ , we may conclude that (12f) is indeed equivalent to (11) under Assumption 4.1. □

Note that the semi-infinite inequality in (12c) generalizes the nonlinear uncertain constraints studied in [1] because it involves an additional norm term and as the loss function $\ell ( \xi )$ is not necessarily concave under Assumption 4.1. As in [1], however, the semi-infinite constraint admits a robust counterpart that involves the conjugate of the loss function and the support function of the uncertainty set.

From the proof of Theorem 4.2 it is immediately clear that the worst-case expectation (1O) is conservatively approximated by the optimal value of the finite convex program (12f) even if Assumption 4.1 fails to hold. In this case the sum $- \ell _ { k } + \chi _ { \Xi }$ in (12f） must be evaluated under our conventions of extended arithmetics,whereby $\infty - \infty = \infty$ . These observations are formalized in the following corollry.

Corollary 4.3 [Approximate convex reduction] For any $\varepsilon \geq 0$ , the worst-case expectation (1O) is smaller or equal to the optimal value of the finite convex program (12f).

# 4.2 Extremal distributions

Stress test experiments are instrumental to assess the quality of candidate decisions in stochastic optimization. Meaningful stress tests require a good understanding of the extremal distributions from within the Wasserstein ball that achieve the worstcase expectation (1O) for various loss functions.We now show that such extremal distributions can be constructed systematically from the solution of a convex program akin to (11).

Theorem 4.4 (Worst-case distributions) IfAssumption 4.1 holds, then the worst-case expectation (1O) coincides with the optimal value of the finite convex program

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { s u p } _ { \sigma _ { i k } , q _ { i k } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } \ell _ { k } ( \widehat { \xi } _ { i } - \frac { q _ { i k } } { \alpha _ { i k } } ) } \\ { \mathrm { s . t . } } & { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \| q _ { i k } \| \leq \varepsilon } \\ { \displaystyle \sum _ { k = 1 } ^ { K } \alpha _ { i k } = 1 } & { \forall i \leq N } \\ { \displaystyle \alpha _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K } \\ { \displaystyle \widehat { \xi } _ { i } - \frac { q _ { i k } } { \alpha _ { i k } } \in \Xi } & { \forall i \leq N , \forall k \leq K } \end{array} \right.
$$

irrespective of $\dot { \boldsymbol { \varepsilon } } \geq 0 . L e t \left\{ \alpha _ { i k } ( r ) , q _ { i k } ( r ) \right\} _ { r \in \mathbb { N } } b e \boldsymbol { \varepsilon }$ a sequence offeasible decisions whose objective values converge to the supremum of (13). Then, the discrete probability distributions

$$
\mathbb { Q } _ { r } { : = } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } ( r ) \delta _ { \xi _ { i k } ( r ) } \quad w i t h \quad \xi _ { i k } ( r ) : = \widehat { \xi } _ { i } - \frac { q _ { i k } ( r ) } { \alpha _ { i k } ( r ) }
$$

belong to the Wassersteinball $\mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ and attain the supremum of (10)asymptotically i.e.，

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ] = \operatorname* { l i m } _ { r  \infty } \mathbb { E } ^ { \mathbb { Q } _ { r } } \big [ \ell ( \xi ) \big ] = \operatorname* { l i m } _ { k  \infty } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } ( r ) \ell \big ( \xi _ { i k } ( r ) \big ) .
$$

We highlight that all fractions in (13) must again be evaluated under our conventions of extended arithmetics. Specifically, if $\alpha _ { i k } = 0$ and $q _ { i k } \neq 0$ ,then $q _ { i k } / \alpha _ { i k }$ has at least one component equal to $+ \infty$ or $- \infty$ , which implies that $\hat { \xi } _ { i } - q _ { i k } / \alpha _ { i k } \notin \Xi$ In contrast, if $\alpha _ { i k } = 0$ and $q _ { i k } = 0$ ,then $\widehat { \xi _ { i } } - q _ { i k } / \alpha _ { i k } = \widehat { \xi _ { i } } \in \Xi$ Moreover, the $i k$ -th component in the objective function of (13) evaluates to O whenever $\alpha _ { i k } = 0$ regardless of $q _ { i k }$

The proof of Theorem 4.4 is based on the following technical lemma.

Lemma 4.5 Define $F : \mathbb { R } ^ { m } \times \mathbb { R } _ { + } \to \overline { { \mathbb { R } } }$ through $\begin{array} { r } { F ( q , \alpha ) = \operatorname* { i n f } _ { z \in \mathbb { R } ^ { m } } \left. z , q - \alpha \widehat { \xi } \right. + } \end{array}$ $\alpha f ^ { * } ( z )$ for some proper, convex, and lower semicontinuous function $f : \mathbb { R } ^ { m }  \overline { { \mathbb { R } } }$ and reference point $\hat { \xi } \in \mathbb { R } ^ { m }$ . Then, $F$ coincides with the (extended) perspective function of the mapping $q \mapsto - f ( \widehat { \xi } - q )$ ,that is,

$$
F ( q , \alpha ) = \left\{ \begin{array} { l l } { - \alpha f \big ( \widehat { \xi } - q / \alpha \big ) } & { i f \alpha > 0 , } \\ { - \chi _ { \{ 0 \} } ( q ) } & { i f \alpha = 0 . } \end{array} \right.
$$

Proof By construction, we have $\begin{array} { r } { F ( q , 0 ) = \operatorname* { i n f } _ { z \in \mathbb { R } ^ { m } } \left. z , q \right. = - \chi _ { \{ 0 \} } ( q ) } \end{array}$ .For $\alpha > 0$ ,on the other hand,the definition of conjugacy implies that

$$
F ( q , \alpha ) = - [ \alpha f ^ { * } ] ^ { * } ( \alpha \widehat { \xi } - q ) = - \alpha [ f ^ { * } ] ^ { * } \widehat { \left( \xi - q / \alpha \right) } .
$$

The claim then follows because $[ f ^ { * } ] ^ { * } = f$ for any proper, convex,and lower semicontinuous function $f$ [4,Proposition 1.6.1(c)]. Additional information on perspective functions can be found in [12, Section 2.2.3, p. 39]. □

Proof of Theorem 4.4 By Theorem 4.2， which applies under Assumption 4.1, the worst-case expectation (1O) coincides with the optimal value of the convex program (11). From the proof of Theorem 4.2 we know that (11) is equivalent to (12f). The Lagrangian dual of (12f) is given by

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { s u p } _ { \beta _ { i k } , \boldsymbol { \alpha } _ { i k } } \displaystyle \operatorname* { i n f } _ { \lambda , \boldsymbol { \delta } _ { i } , \boldsymbol { z } _ { i k } } \lambda _ { \mathcal { S } } + \sum _ { i = 1 } ^ { N } \left[ \frac { s _ { i } } { N } + \sum _ { k = 1 } ^ { K } \left[ \beta _ { i k } \left( \| \boldsymbol { z } _ { i k } \| _ { \mathfrak { s } } - \lambda \right) + \alpha _ { i k } \left( [ - \ell _ { k } + \chi _ { \perp } ] ^ { * } ( \boldsymbol { z } _ { i k } ) - \left. \boldsymbol { z } _ { i k } , \widehat { \xi } _ { i } \right. - s _ { i } \right) \right] \right] } \\ { \displaystyle \operatorname* { s . t } _ { \boldsymbol { \alpha } _ { i k } } \sum _ { \boldsymbol { \alpha } _ { i k } } \sum _ { 0 } 0 } & { \forall i \leq N , \forall \boldsymbol { k } \leq K } \\ { \displaystyle \beta _ { i k } \geq 0 } & { \forall i \leq N , \forall \boldsymbol { k } \leq K , } \end{array} \right.
$$

where the products of dual variables and constraint functions in the objective are evaluated under the standard convention $0 \cdot \infty = 0$ . Strong duality holds since the function $[ - \ell _ { k } + \chi _ { \Xi } ] ^ { * }$ is proper, convex, and lower semicontinuous under Assumption 4.1 and because this function appears in a constraint of (12f) whose right-hand side is a free decision variable. By explicitly carrying out the minimization over $\lambda$ and $s _ { i }$ , one can show that the above dual problem is equivalent to

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { s u p } _ { \beta _ { i k } , \alpha _ { i k } } \displaystyle \operatorname* { i n f } _ { z i k } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \beta _ { i k } \| z _ { i k } \| _ { * } + \alpha _ { i k } [ - \ell _ { k } + \chi _ { \Xi } ] ^ { * } ( z _ { i k } ) - \alpha _ { i k } \big \langle z _ { i k } , \widehat { \xi } _ { i } \big \rangle } \\ { \displaystyle \mathit { s . c . } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \beta _ { i k } = \varepsilon } \\ { \displaystyle \qquad \sum _ { k = 1 } ^ { K } \alpha _ { i k } = \frac { 1 } { N } } & { \forall i \le N } \\ { \displaystyle \qquad \alpha _ { i k } \ge 0 } & { \forall i \le N , \forall k \le K } \\ { \displaystyle \beta _ { i k } \ge 0 } & { \forall i \le N , \forall k \le K . } \end{array} \right.
$$

By using the definition of the dual norm, (14a) can be re-expressed as

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { s u p } _ { \beta _ { i k } , \alpha _ { i k } } \displaystyle \operatorname* { i n f } _ { z _ { i k } \leq 1 } \displaystyle \sum _ { i = 1 } ^ { N } \displaystyle \operatorname* { m a x } _ { i = 1 } \left\{ z _ { i k } , q _ { i k } \right\} + \alpha _ { i k } [ - \ell _ { k } + \chi _ { \Xi } ] ^ { * } ( z _ { i k } ) - \alpha _ { i k } \big \langle z _ { i k } , \widehat { \xi } _ { i } \big \rangle \displaystyle ] } \\ { \displaystyle \mathrm { s . t . } \quad \displaystyle \sum _ { i = 1 } ^ { N } \displaystyle \sum _ { k = 1 } ^ { K } \beta _ { i k } = \varepsilon } \\ { \displaystyle \sum _ { k = 1 } ^ { K } \alpha _ { i k } = \frac { 1 } { N } } & { \forall i \leq N } \\ { \displaystyle \alpha _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K } \\ { \displaystyle \beta _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K } \end{array} \right.
$$

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \underset { \beta _ { i k } , \alpha _ { i k } } { \operatorname* { s u p } } \underset { \| \boldsymbol { q } _ { i k } \| \leq \beta _ { i k } } { \operatorname* { m a x } } \underset { \boldsymbol { i } = 1 } { \operatorname* { i n f } } \underset { \boldsymbol { i } = 1 } { \overset { N } { \sum } } \left. \boldsymbol { z } _ { i k } , \boldsymbol { q } _ { i k } \right. + \alpha _ { i k } [ - \boldsymbol { \ell } _ { k } + \chi \boldsymbol { \Xi } ] ^ { * } ( \boldsymbol { z } _ { i k } ) - \alpha _ { i k } \big \langle \boldsymbol { z } _ { i k } , \widehat { \boldsymbol { \xi } } _ { i } \big \rangle } \\ { \underset { \mathrm { s . t . } } { = } \underset { \ i = 1 } { \overset { N } { \sum } } \underset { \mathrm { - } 1 } { \overset { K } { \sum } } \underset { \boldsymbol { k } = 1 } { \overset { \sum } { \sum } } \beta _ { i k } = \varepsilon } \\ { \underset { \underset { \boldsymbol { k } = 1 } { \overset { K } { \sum } } } \alpha _ { i k } = \frac { 1 } { N } } & { \forall i \leq N } \\ { \alpha _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K } \\ { \beta _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K , } \end{array} \right. } \end{array}
$$

where (14c) follows from the classical minimax theorem and the fact that the $q _ { i k }$ variables range over a non-empty and compact feasible set for any finite $\varepsilon$ ; see [4, Proposition 5.5.4]. Eliminating the $\beta _ { i k }$ variables and using Lemma 4.5 allows us to reformulate (14c) as

$$
\begin{array} { r l } &  \left\{ \begin{array} { l l } { \underset { \mathrm { c r o s t } } { \operatorname* { s u p } } \ \underset { n = 1 } { \overset { N } { \sum } } \ \underset { i = 1 } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { i = 1 } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \right\} } \\ { \underset { \mathrm { c r o s t } } { \overset { \sum } { \sum } } \ \ \underset { i = 1 } { \overset { N } { \sum } } \ \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \ \underset { \mathrm { d } \leq i } { \overset { N } { \sum } } \ \ } & { \mathrm { d } \mathrm { d } } \\ { \ } &  \underset { \mathrm { c r o s t } } { \overset { \sum } { \sum } } \ \underset { \mathrm { d } \geq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \geq i } { \overset { N } { \sum } } \ \ \underset { \mathrm { d } \geq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \geq i } { \overset { N } { \sum } } \ \ \underset { \mathrm { d } \geq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \geq i } { \overset { N } { \sum } } \ \underset { \mathrm { d } \geq i } { \overset { N } { \sum } } \ \ \end{array} \end{array}
$$

Our conventions of extended arithmetics imply that the $i k$ -th term in the objective function of problem (14e) simplifies to

$$
\alpha _ { i k } \ell _ { k } \biggl ( \widehat { \xi } _ { i } - \frac { q _ { i k } } { \alpha _ { i k } } \biggr ) - \chi _ { \Xi } \biggl ( \widehat { \xi } _ { i } - \frac { q _ { i k } } { \alpha _ { i k } } \biggr ) .
$$

Indeed, for $\alpha _ { i k } > 0$ ,this identity trivially holds.For $\alpha _ { i k } = 0$ ,on the other hand, the $i k$ -th objective term in (14e) reduces to $- \chi _ { \{ 0 \} } ( q _ { i k } )$ . Moreover, the first term in (14f) vanishes whenever $\alpha _ { i k } = 0$ regardless of $q _ { i k }$ , and the second term in (14f) evaluates to 0if $q _ { i k } = 0$ (as $0 / 0 = 0$ and $\bar { \xi } _ { i } \in \Xi )$ and to $- \infty$ if $q _ { i k } \neq 0$ (as $q _ { i k } / 0$ has at least one infinite component, implying that $\widehat { \xi } _ { i } + q _ { i k } / 0 \notin \Xi )$ . Therefore,(14f) also reduces to $- \chi _ { \{ 0 \} } ( q _ { i k } )$ when $\alpha _ { i k } = 0$ . This proves that the $i k$ -th objective term in (14e) coincides with (14f). Substituting (14f) into (14e) and re-expressing $\begin{array} { r } { - \chi _ { \Xi } ( \widehat { \xi _ { i } } - \frac { q _ { i k } } { \alpha _ { i k } } ) } \end{array}$ in terms of an explicit hard constraint yields

$$
\{ \begin{array} { l l } { \displaystyle \operatorname* { s u p } _ { \alpha _ { i k } , q _ { i k } } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } \ell _ { k } \widehat { \xi } _ { i } - \frac { q _ { i k } } { \alpha _ { i k } } \} } \\ { \mathrm { ~ s . t . ~ } } \\ { \displaystyle \quad \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \| q _ { i k } \| \leq \varepsilon } \\ { \displaystyle \qquad \sum _ { k = 1 } ^ { K } \alpha _ { i k } = \frac { 1 } { N } \qquad } & { \forall i \leq N } \\ { \displaystyle \qquad \alpha _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K } \\ { \displaystyle \widehat { \xi } _ { i } - \frac { q _ { i k } } { \alpha _ { i k } } \in \Xi } & { \forall i \leq N , \forall k \leq K . } \end{array} 
$$

Finally, replacing $\left\{ \alpha _ { i k } , q _ { i k } \right\}$ with $\textstyle { \frac { 1 } { N } } \left\{ \alpha _ { i k } , q _ { i k } \right\}$ shows that $( 1 4 \mathrm { g } )$ is equivalent to (13).   
This completes the first part of the proof.

As for the second claim, let $\{ \alpha _ { i k } ( r ) , q _ { i k } ( r ) \} _ { r \in \mathbb { N } }$ be a sequence of feasible solutions that attains thesupremum in (13),andset $\begin{array} { r } { \xi _ { i k } ( r ) : = \widehat { \xi } _ { i } - \frac { q _ { i k } ( r ) } { \alpha _ { i k } ( r ) } \in \Xi } \end{array}$ Then, the discrete distribution

$$
\Pi _ { r } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } ( r ) \delta _ { \left( \xi _ { i k } ( r ) , \widehat { \xi } _ { i } \right) }
$$

has the distribution $\mathbb { Q } _ { r }$ defined in the theorem statement and the empirical distribution ${ \widehat { \mathbb { P } } } _ { N }$ as marginals.By the definition of the Wasserstein metric, $\Pi _ { r }$ represents a feasible mass transportation plan that provides an upper bound on the distance between ${ \widehat { \mathbb { P } } } _ { N }$ and $\mathbb { Q } _ { r }$ ; see Definition 3.1. Thus, we have

$$
\begin{array} { r l r } {  { d _ { \mathbb { W } } \big ( \mathbb { Q } _ { r } , \widehat { \mathbb { P } } _ { N } \big ) \le \int _ { \Xi ^ { 2 } } \| \xi - \xi ^ { \prime } \| \ \Pi _ { r } ( { \mathrm { d } } \xi , { \mathrm { d } } \xi ^ { \prime } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } ( r ) \| \xi _ { i k } ( r ) \| } } \\ & { } & { - \widehat { \xi } _ { i } \| = \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \displaystyle \sum _ { k = 1 } ^ { K } \| q _ { i k } ( r ) \| \le \varepsilon , } \end{array}
$$

where the last inequality follows readily from the feasibility of $q _ { i k } ( r )$ in (13). We conclude that

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ] \geq \operatorname* { l i m } _ { k  \infty } \mathbb { E } ^ { \mathbb { Q } _ { r } } \big [ \ell ( \xi ) \big ] = \displaystyle \operatorname* { l i m } _ { k  \infty } \operatorname* { s u p } _ { k } \frac { 1 } { \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } } \alpha _ { i k } ( r ) \ell \big ( \xi _ { i k } ( r ) \big ) } & { } \\ { \displaystyle \geq \operatorname* { l i m } _ { k  \infty } \operatorname* { s u p } _ { N } \frac { 1 } { \displaystyle \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } } \alpha _ { i k } ( r ) \ell _ { k } \big ( \xi _ { i k } ( r ) \big ) = \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ] . } \end{array}
$$

where the first inequality holds as $\mathbb { Q } _ { r } \ \in \ \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ for all $k \in \mathbb N$ ，and the second inequality uses the trivial estimate $\ell \geq \ell _ { k }$ for all $k \leq K$ . The last equality follows from the construction of $\alpha _ { i k } ( r )$ and $\xi _ { i k } ( r )$ and the fact that (13) coincides with the worst-case expectation (10). □

In the rest of this section we discuss some notable properties of the convex program (13).

In the ambiguity-free limit, that is, when the radius of the Wasserstein ball is set to Zero, then the optimal value of the convex program (13) reduces to the expected loss under the empirical distribution.Indeed, for $\varepsilon = 0$ all $q _ { i k }$ variables are forced to zero, and $\alpha _ { i k }$ enters the objective only through $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \alpha _ { i k } = \frac { 1 } { N } } \end{array}$ .Thus,the objective function of (13) simplifies to $\mathbb { E } ^ { \widehat { \mathbb { P } } _ { N } } [ \ell ( \xi ) ]$

We further emphasize thatit is not possible to guarantee the existence of a worst-case distribution that attains the supremum in (1O). In general, as shown in Theorem 4.4, we can only construct a sequence of distributions that attains the supremum asymptotically. The following example discusses an instance of (1O) that admits no worst-case distribution.

![](images/3adf30c260c7972309c57fbe20a82020f575167dab00ff32885b5cf474105da4.jpg)  
Fig.1 Example of a worst-case expectation problem without a worst-case distribution

Example 2 (Non-existence of a worst-case distribution) Assume that $\Xi = \mathbb { R }$ ， $N = 1$ ， $\widehat { \xi } _ { 1 } = \mathrm { { 0 } }$ ， $K = 2$ ， $\ell _ { 1 } ( \xi ) = 0$ and $\ell _ { 2 } ( \xi ) = \xi - 1 .$ . In this case we have $\widehat { \mathbb { P } } _ { N } = \delta _ { \{ 0 \} }$ ,and problem (13) reduces to

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \delta _ { 0 } ) } \mathbb { E } ^ { \mathbb { Q } } \bigl [ \ell ( \xi ) \bigr ] = \left\{ \begin{array} { l l } { \operatorname* { s u p } _ { \alpha _ { 1 j } , q _ { 1 j } } - q _ { 1 2 } - \alpha _ { 1 2 } } \\ { \mathrm { ~ s . t . ~ } | q _ { 1 1 } | + | q _ { 1 2 } | \le \varepsilon } \\ { \alpha _ { 1 1 } + \alpha _ { 1 2 } = 1 } \\ { \alpha _ { 1 1 } \ge 0 , ~ \alpha _ { 1 2 } \ge 0 . } \end{array} \right.
$$

The supremum on the right-hand side amounts to $\varepsilon$ and is attained, for instance,by the sequence $\alpha _ { 1 1 } ( r ) = 1 - \stackrel { \triangledown } { _ { k } } , \alpha _ { 1 2 } ( r ) = \textstyle { \frac { 1 } { k } } , q _ { 1 1 } ( r ) = 0 , q _ { 1 2 } ( r ) = - \varepsilon$ for $k \in \mathbb N$ .Define

$$
\mathbb { Q } _ { r } = \alpha _ { 1 1 } ( r ) \delta _ { \xi _ { 1 1 } ( r ) } + \alpha _ { 1 2 } ( r ) \delta _ { \xi _ { 1 2 } ( r ) } ,
$$

with {11(r）=1-（） and $\begin{array} { r } { \xi _ { 1 2 } ( r ) = \widehat { \xi } _ { 1 } - \frac { q _ { 1 2 } ( r ) } { \alpha _ { 1 2 } ( r ) } = \varepsilon k } \end{array}$ . By Theorem 4.4, the two-point distributions $\mathbb { Q } _ { r }$ reside within the Wasserstein ball of radius $\varepsilon$ around $\delta _ { 0 }$ and asymptotically attain the supremum in the worst-case expectation problem. However, this sequence has no weak limit as $\xi _ { 1 2 } ( r ) = \varepsilon k$ tends to infinity, see Fig. 1. In fact, no single distribution can attain the worst-case expectation. Assume for the sake of contradiction that there exists $\mathbb { Q } ^ { \star } \in \mathbb { B } _ { \varepsilon } ( \delta _ { 0 } )$ with $\mathbb { E } ^ { \mathbb { Q } ^ { \hat { \star } } } [ \ell ( \xi ) ] = \varepsilon$ . Then, we find $\varepsilon = \mathbb { E } ^ { \mathbb { Q } ^ { \star } } [ \ell ( \xi ) ] < \mathbb { E } ^ { \mathbb { Q } ^ { \star } } [ | \xi | ] \leq \varepsilon$ ,where the strict inequality follows from the relation $\ell ( \xi ) < | \xi |$ for all $\xi \neq 0$ and the observation that $\mathbb { Q } ^ { \star } \neq \delta _ { 0 }$ , while the second inequality follows from Theorem 3.2. Thus, $\mathbb { Q } ^ { \star }$ does not exist.

The existence of a worst-case distribution can, however, be guaranteed in some special cases.

Corollary 4.6 (Existence of a worst-case distribution) Suppose that Assumption 4.1 holds. If the uncertainty set $\Xi$ is compact or the loss function is concave (i.e, $K = 1 .$ ）， then the sequence $\{ \alpha _ { i k } ( r ) , \xi _ { i k } ( r ) \} _ { r \in \mathbb { N } }$ constructed in Theorem 4.4 has an accumulation point $\{ \alpha _ { i k } ^ { \star } , \xi _ { i k } ^ { \star } \}$ ,and

![](images/83afcdde9b646ae04515dd6186a444efb3b7faad8b178f3c1a31a61981a8d4f9.jpg)  
Fig.2 Representative distributions in balls centered at ${ \widehat { \mathbb { P } } } _ { N }$ induced by different metrics.(a) Empirical distribution on a training dataset with $N = 2$ samples.(b) A representative discrete distribution in the total variation or the Kullback-Leiber ball (c)A representative discrete distribution in the Wasserstein ball

$$
\mathbb { Q } ^ { \star } { : = } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } ^ { \star } \delta _ { \xi _ { i k } ^ { \star } }
$$

is a worst-case distribution achieving the supremum in (10).

Proof If $\Xi$ is compact, then the sequence $\{ \alpha _ { i k } ( r ) , \xi _ { i k } ( r ) \} _ { r \in \mathbb { N } }$ has a converging subsequence with limit $\{ \alpha _ { i k } ^ { \star } , \xi _ { i k } ^ { \star } \}$ Similarly, if $K = 1$ then $\alpha _ { i 1 } = 1$ for all $i \leq N$ , in which case (13） reduces to a convex optimization problem with an upper semicontinuous objective function over a compact feasible set. Hence, its supremum is attained at a point $\{ \alpha _ { i k } ^ { \star } , \xi _ { i k } ^ { \star } \}$ . In both cases, Theorem 4.4 guarantees that the distribution $\mathbb { Q } ^ { \star }$ implied by $\{ \alpha _ { i k } ^ { \star } , \xi _ { i k } ^ { \star } \}$ achieves the supremum in (10). □

The worst-case distribution of Corollary4.6 is discrete, and its atoms $\xi _ { i k } ^ { \star }$ reside in the neighborhood of the given data points $\widehat { \xi } _ { i }$ . By the constraints of problem (13), the probability-weighted cumulative distance between the atoms and the respective data points amounts to

$$
\sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \alpha _ { i k } \| \xi _ { i k } ^ { \star } - \widehat { \xi } _ { i } \| = \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \| q _ { i k } \| \leq \varepsilon ,
$$

which is bounded above by the radius of the Wasserstein ball. The fact that the worst-case distribution $\mathbb { Q } ^ { \star }$ (if it exists) is supported outside of $\widehat { \Xi } _ { N }$ is a key feature distinguishing the Wasserstein ball from the ambiguity sets induced by other probability metrics such as the total variation distance or the Kullback-Leibler divergence; see Fig.2. Thus, the worst-case expectation criterion based on Wasserstein balls advocated in this paper should appeal to decision makers who wish to immunize their optimization problems against perturbations of the data points.

Remark 4.7 (Weak coupling) We highlight that the convex program (13) is amenable to decomposition and parallelization techniques as the decision variables associated with different sample points are only coupled through the norm constraint. We expect the resulting scenario decomposition to offer a substantial speedup of the solution times for problems involving large datasets. Efficient decomposition algorithms that could be used for solving the convex program (13) are described, for example,in [35] and [5, Chapter 4].

# 5 Special loss functions

We now demonstrate that the convex optimization problems （11） and （13） reduce to computationally tractable conic programs for several loss functions of practical interest.

# 5.1 Piecewise affine loss functions

We first investigate the worst-case expectations of convex and concave piecewise affine loss functions,which arise,for example, in option pricing [8], risk management [34] and in generic two-stage stochastic programming [6]. Moreover, piecewise affne functions frequently serve as approximations of smooth convex or concave loss functions.

Corollary 5.1 (Piecewise affine loss functions) Suppose that the uncertainty set is a polytope, that is, $\Xi = \{ \xi \in \mathbb { R } ^ { m } : C \xi \leq d \}$ where $C$ is a matrix and d a vector of appropriate dimensions. Moreover, consider the affne functions $a _ { k } ( \xi ) : = \left. { a _ { k } , \xi } \right. + b _ { k }$ for all $k \leq K$

(i）If $\begin{array} { r } { \ell ( \xi ) = \operatorname* { m a x } _ { k \le K } a _ { k } ( \xi ) . } \end{array}$ ， then the worst-case expectation (10) evaluates to

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { ~ \displaystyle \operatorname* { i n f } _ { \lambda , s _ { i } , \gamma _ { i k } } ~ \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \mathrm { ~ s . t . ~ } ~ b _ { k } + \left. a _ { k } , \widehat { \xi } _ { i } \right. + \left. \gamma _ { i k } , d - C \widehat { \xi } _ { i } \right. \leq s _ { i } } & { \forall i \leq N , \forall k \leq K } \\ { ~ \| C ^ { \top } \gamma _ { i k } - a _ { k } \| _ { * } \leq \lambda } & { \forall i \leq N , \forall k \leq K } \\ { ~ \gamma _ { i k } \geq 0 ~ } & { \forall i \leq N , \forall k \leq K . } \end{array} \right. } \end{array}
$$

(ii） If $\begin{array} { r } { \ell ( \xi ) = \operatorname* { m i n } _ { k \le K } { a _ { k } ( \xi ) } } \end{array}$ , then the worst-case expectation (10) evaluates to

$$
\left\{ \begin{array} { c l l } { \displaystyle { \operatorname* { i n f } _ { { { \dot { x } } } , s _ { i } , \gamma _ { i } , \theta _ { i } } \lambda _ { \varepsilon } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } } & & \\ { \mathrm { ~ s . t . ~ } } & { \displaystyle { \langle \theta _ { i } , b + A { \widehat { \xi } } _ { i } \rangle + \big \langle \gamma _ { i } , d - C widehat { \xi } _ { i } \big \rangle \leq s _ { i } } } & { \forall i \leq N } \\ { \displaystyle } & { \| C ^ { \top } \gamma _ { i } - A ^ { \top } \theta _ { i } \| _ { * } \leq \lambda } & { \forall i \leq N } \\ { \displaystyle } & { \big \langle \theta _ { i } , e \big \rangle = 1 } & { \forall i \leq N } \\ { \displaystyle } & { \gamma _ { i } \geq 0 } & { \forall i \leq N } \\ { \theta _ { i } \geq 0 } & { \forall i \leq N , } \end{array} \right.
$$

where $A$ is the matrix with rows $a _ { k } ^ { \mathsf { T } }$ ， $k \leq K$ ， $b$ is the column vector with entries $b _ { k }$ $, k \leq K$ ,and $e$ is the vector of all ones.

Proof Assertion (i) is an immediate consequence of Theorem 4.2, which applies because $\ell ( x )$ is the pointwise maximum of the affine functions $\ell _ { k } ( \xi ) = a _ { k } ( \xi ) , k \le K$ ， and thus Assumption 4.1 holds for $J = K$ . By definition of the conjugacy operator,

we have

$$
[ - \ell _ { k } ] ^ { * } ( z ) = [ - a _ { k } ] ^ { * } ( z ) = \operatorname* { s u p } _ { \xi } \left. z , \xi \right. + \left. a _ { k } , \xi \right. + b _ { k } = { \left\{ \begin{array} { l l } { b _ { k } } & { { \mathrm { i f ~ } } z = - a _ { k } , } \\ { \infty } & { { \mathrm { e l s e } } , } \end{array} \right. }
$$

and

$$
\begin{array} { r } { \sigma _ { \Xi } ( \nu ) = \left\{ \begin{array} { l l } { \underset { \xi } { \operatorname* { s u p } } \left. \nu , \xi \right. } \\ { \mathrm { s . t . } C \xi \leq d } \end{array} \right. = \left\{ \begin{array} { l l } { \underset { \gamma \geq 0 } { \operatorname* { i n f } } \left. \gamma , d \right. } \\ { \mathrm { s . t . } C ^ { \intercal } \gamma = \nu , } \end{array} \right. } \end{array}
$$

where the last equality follows from strong duality, which holds as the uncertainty set is non-empty. Assertion (i) then follows by substituting the above expressions into (11).

Assertion (ii) also follows directly from Theorem 4.2 because $\ell ( \xi ) = \ell _ { 1 } ( \xi ) =$ $\mathrm { m i n } _ { k \le K } { { a } _ { j } } ( \xi )$ is concave and thus satisfies Assumption 4.1 for $J = 1$ . In this setting, we find

$$
- \ell ] ^ { * } ( z ) = \operatorname* { s u p } _ { \xi } \left. z , \xi \right. + \operatorname* { m i n } _ { k \le K } \left\{ \left. a _ { k } , \xi \right. + b _ { k } \right\} = \left\{ \begin{array} { l } { \operatorname* { s u p } \left. z , \xi \right. + \tau } \\ { \xi , \tau } \\ { \mathrm { s . t . } A \xi + b \ge \tau e } \end{array} \right. = \left\{ \begin{array} { l } { \operatorname* { i n f } \left. \theta , b \right. } \\ { \theta \ge 0 } \\ { \mathrm { s . t . } A ^ { \tau } \theta = - z \in \pi } \\ { \left. \theta , e \right. = 1 } \end{array} \right. \mathrm { ~ a ~ n ~ d ~ } \tau = 1 \mathrm { ~ }
$$

where the last equality follows again from strong linear programming duality, which holds since the primal maximization problem is feasible. Assertion (ii) then follows by substituting $[ - \ell ] ^ { * }$ as well as the formula for $\sigma _ { \Xi }$ from the proof of assertion (i) into (11). □

As a consistency check,we ascertain that in the ambiguity-free limit, the optimal value of (15a) reduces to the expectation of $\operatorname* { m a x } _ { k \leq K } { a _ { k } ( \xi ) }$ under the empirical distribution. Indeed, for $\varepsilon = 0$ , the variable $\lambda$ can be set to any positive value at no penalty. For this reason and because all training samples must belong to the uncertainty set (i.e., $d - C \widehat { \xi _ { i } } \geq 0$ for all $i \ \leq \ N .$ ), it is optimal to set $\gamma _ { i k } = 0$ .This in turn implies that $s _ { i } = \operatorname* { m a x } _ { k \le K } a _ { k } ( \widehat { \xi } _ { i } )$ at optimality,inhichase ∑i=1 Si represents the sample average of the convex loss function at hand.

An analogous argument shows that, for $\varepsilon = 0$ , the optimal value of (15b) reduces to the expectation of $\mathrm { m i n } _ { k \le K } { a _ { k } ( \xi ) }$ under the empirical distribution.As before, $\lambda$ can be increased at no penalty. Thus, we conclude that $\gamma _ { i } = 0$ and

$$
s _ { i } = \operatorname* { m i n } _ { \theta _ { i } \geq 0 } \left\{ \left. \theta _ { i } , b + A { \widehat { \xi } } _ { i } \right. : \left. \theta _ { i } , e \right. = 1 \right\} = \operatorname* { m i n } _ { k \leq K } a _ { k } ( { \widehat { \xi } } _ { i } )
$$

atoptimalityi hich case $\textstyle { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } s _ { i }$ is hlti function.

# 5.2 Uncertainty quantification

A problem of great practical interest is to ascertain whether a physical, economic or engineering system with an uncertain state $\xi$ satisfies a number of safety constraints with high probability. In the following we denote by A the set of states in which the system is safe. Our goal is to quantify the probability of the event $\xi \in \mathbb { A }$ $( \xi \notin \mathbb { A } )$ under an ambiguous state distribution that is only indirectly observable through a finite training dataset. More precisely, we aim to calculate the worst-case probability of the system being unsafe, i.e.,

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \left[ \xi \notin \mathbb { A } \right] ,
$$

as wellas the best-case probability of the system being safe, that is,

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( { \widehat { \mathbb { P } } } _ { N } ) } \mathbb { Q } \left[ \xi \in { \mathbb { A } } \right] .
$$

Remark 5.2 (Data-dependent sets） The set A may even depend on the samples $\widehat { \xi } _ { 1 } , \dots , \widehat { \xi } _ { N }$ , in which case A is renamed as $\widehat { \mathbb { A } }$ If the Wasserstein radius $\varepsilon$ is set to $\varepsilon _ { N } ( \beta )$ , then we have $\mathbb { P } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ with probability $1 \_ \beta$ , implying that (16a) and (16b） still provide $1 - \beta$ confidence bounds on $\mathbb { P } [ \xi \notin \widehat { \mathbb { A } } ]$ and $\mathbb { P } [ \boldsymbol { \xi } \in \widehat { \mathbb { A } } ]$ , respectively.

Corollary 5.3 (Uncertainty quantification) Suppose that the uncertainty set is a polytope of the form $\Xi = \{ \xi \in \mathbb { R } ^ { m } : C \xi \leq d \}$ as in Corollary 5.1.

(i) $I f \mathbb { A } = \{ \xi \in \mathbb { R } ^ { m } : A \xi < b \}$ is an open polytope and the halfspace $\left\{ \xi : \left. a _ { k } , \xi \right. \geq \right.$ $b _ { k } \bigg \}$ has a nonempty intersection with $\Xi$ for any $k \ \leq \ K$ ，where $a _ { k }$ is the $k$ -th row of the matrix $A$ and $b _ { k }$ is the $k$ -th entry of the vector $b$ ， then the worst-case probability (16a) is given by

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { { \scriptstyle \lambda , s _ { i } , \gamma _ { i k } , \theta _ { i k } } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \mathrm { ~ s . t . ~ } } & { 1 - \theta _ { i k } \left( b _ { k } - \left. a _ { k } , \widehat { { \xi } _ { i } } \right. \right) + \left. { \gamma _ { i k } , d - C \widehat { { \xi } _ { i } } } \right. \leq s _ { i } \quad \forall i \leq N , \forall k \leq K } \\ & { \| a _ { k } \theta _ { i k } - C ^ { \top } \gamma _ { i k } \| _ { * } \leq \lambda } & { \forall i \leq N , \forall k \leq K } \\ & { \gamma _ { i k } \succeq 0 } & { \forall i \leq N , \forall k \leq K } \\ & { \theta _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K } \\ & { s _ { i } \geq 0 } & { \forall i \leq N . } \end{array} \right.
$$

(i $) ~ I f \mathbb { A } = \{ \xi \in \mathbb { R } ^ { m } : A \xi \leq b \}$ is a closed polytope that has a nonempty intersection with E, then the best-case probability (l6b) is given by

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { \boldsymbol { \lambda } , \boldsymbol { s } _ { i } , \boldsymbol { \gamma } _ { i } , \boldsymbol { \theta } _ { i } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \mathrm { ~ s . t . ~ } } & { 1 + \left. \boldsymbol { \theta } _ { i } , \boldsymbol { b } - \boldsymbol { A } \widehat { \xi } _ { i } \right. + \left. \boldsymbol { \gamma } _ { i } , d - C \widehat { \xi } _ { i } \right. \leq s _ { i } } \\ & { \| \boldsymbol { A } ^ { \intercal } \boldsymbol { \theta } _ { i } + C ^ { \intercal } \boldsymbol { \gamma } _ { i } \| _ { * } \leq \boldsymbol { \lambda } } & { \forall i \leq N } \\ & { \boldsymbol { \gamma } _ { i } \geq 0 } & { \forall i \leq N } \\ & { \boldsymbol { \theta } _ { i } \geq 0 } & { \forall i \leq N } \\ & { \boldsymbol { s } _ { i } \geq 0 } & { \forall i \leq N . } \end{array} \right.
$$

Proof The uncertainty quantification problems (16a) and (16b) can be interpreted as instances of (1O) with loss functions $\ell = 1 - 1 _ { \mathbb { A } }$ and $\ell = \mathbb { 1 } _ { \mathbb { A } }$ , respectively. In order to be able to apply Theorem 4.2, we should represent these loss functions as finite maxima of concave functions as shown in Fig. 3.

Formally, assertion (i) follows from Theorem 4.2 for a loss function with $K + 1$ pieces if we use the following definitions. For every $k \leq K$ we define

$$
\ell _ { k } ( \xi ) = \left\{ { \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } \left. a _ { k } , \xi \right. \geq b _ { k } , } \\ { - \infty } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.
$$

Moreover, we define $\ell _ { K + 1 } ( \xi ) = 0$ . As illustrated in Fig. 3a, we thus have $\ell ( \xi ) =$ $\mathrm { m a x } _ { k \le K + 1 } \ell _ { k } ( \xi ) = 1 - \mathbb { 1 } _ { \mathbb { A } } ( \xi )$ and

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \left[ \xi \notin \mathbb { A } \right] \ = \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \left[ \ell ( \xi ) \right] .
$$

Assumption 4.1 holds due to the postulated properties of $\mathbb { A }$ and $\Xi$ . In order to apply Theorem 4.2, we must determine the support function $\sigma _ { \Xi }$ ，which is already known from Corollary 5.1, as wellas the conjugate functions of $- \ell _ { k } , k \le K + 1 .$ A standard duality argument yields

$$
[ - \ell _ { k } ] ^ { * } ( z ) = \{ { \begin{array} { l } { \operatorname* { s u p }  z , \xi  + 1 } \\ { \xi } \\ { { \mathrm { s . t . ~ } }  a _ { k } , \xi  \geq b _ { k } } \end{array} } = \{ { \begin{array} { l } { \operatorname* { i n f } _ { \theta \geq 0 } 1 - b _ { k } \theta } \\ { \theta { \geq } 0 } \\ { { \mathrm { s . t . ~ } } a _ { k } \theta = - z , } \end{array} } 
$$

for all $k \leq K$ . Moreover, we have $[ - \ell _ { K + 1 } ] ^ { * } = 0$ if $\xi = 0 ; = \infty$ otherwise. Assertion (ii) then follows by substituting the formulas for $[ - \ell _ { k } ] ^ { * }$ ， $k \leq K + 1$ ,and $\sigma _ { \Xi }$ into (11).

Assertion (ii) follows from Theorem 4.2 by setting $K = 2$ ， $\ell _ { 1 } ( \xi ) = 1 - \chi _ { \mathbb { A } } ( \xi )$ and $\ell _ { 2 } ( \xi ) = 0$ . As illustrated in Fig.3b, this implies that $\ell ( \xi ) = \operatorname* { m a x } \{ \ell _ { 1 } ( \xi ) , \ell _ { 2 } ( \xi ) \} =$ $\mathbb { 1 } _ { \mathbb { A } } ( \xi )$ and

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \left[ \xi \in \mathbb { A } \right] \ = \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \left[ \ell ( \xi ) \right] .
$$

Assumption 4.1 holds by our assumptions on $\mathbb { A }$ and $\Xi$ . In order to apply Theorem 4.2, we thus have to determine the support function $\sigma _ { \Xi }$ , which was already calculated in

![](images/f936607f246b41b00355022c90870405198a4f9b26849b082283c76c269063c8.jpg)  
Fig.3 Representing the indicator function of a convex set and its complement as a pointwise maximum of concave functions.(a) Indicator function of the unsafe set.(b) Indicator function of the safe set

Corollary 5.1, and the conjugate functions of $- \ell _ { 1 }$ and $- \ell _ { 2 }$ . By the definition of the conjugacy operator, we find

$$
[ - \ell _ { 1 } ] ^ { * } ( z ) = \operatorname* { s u p } _ { \xi \in \hat { \mathbb { A } } } \langle z , \xi \rangle + 1 = { \left\{ \begin{array} { l l } { \operatorname* { s u p } _ { \xi } \langle z , \xi \rangle + 1 } \\ { \xi } \\ { { \mathrm { s . t . ~ } } A \xi \leq b } \end{array} \right. } = { \left\{ \begin{array} { l l } { { \operatorname* { i n f } _ { \theta , b } } \langle \theta , b \rangle + 1 } \\ { \theta _ { k \geq 0 } } \\ { { \mathrm { s . t . ~ } } A ^ { \intercal } \theta = z } \end{array} \right. }
$$

where the last equality follows from strong linear programming duality, which holds as the safe set is non-empty. Similarly, we find $[ - \ell _ { 2 } ] ^ { * } = 0$ if $\xi = 0 ; = \infty$ otherwise. Assertion (ii) then follows by substituting the above expressions into (11). □

In the ambiguity-free limit (i.e., for $\varepsilon = 0$ ) the optimal value of (17a) reduces to the fraction of training samples residing outside of the open polytope $\mathbb { A } = \{ \xi : A \xi < b \}$ . Indeed,in this case the variable $\lambda$ can be set to any positive value at no penalty. For this reason and because all training samples belong to the uncertainty set $( \widehat { i . e . , d - C \xi _ { i } } \geq 0$ for all $i \le N )$ , it is optimal to set $\gamma _ { i k } = 0$ If the $i$ -th training sample belongs to A (i.e., $b _ { k } - \left. a _ { k } , \widehat { \xi _ { i } } \right. > 0$ for all $k \leq K ,$ ),then $\theta _ { i k } \geq 1 / ( b _ { k } - \left. a _ { k } , \widehat { \xi } _ { i } \right. )$ for all $k \leq K$ and $s _ { i } = 0$ at optimality. Conversely, if the $i$ -th training sample belongs to the complement of $\mathbb { A }$ ， (i.e., $b _ { k } - \left. \widetilde { a } _ { k } , \widehat { \xi } _ { i } \right. \leq 0$ $\textstyle \sum _ { i = 1 } ^ { N } s _ { i }$ for some coa $k \leq K$ ), then $\theta _ { i k } = 0$ for some $k \leq K$ and $s _ { i } = 1$ at $\varepsilon = 0$ (17b) reduces to the fraction of training samples residing inside of the closed polytope $\mathbb { A } = \{ \xi : A \xi \leq b \}$

# 5.3 Two-stage stochastic programming

A major challenge in linear two-stage stochastic programming is to evaluate the expected recourse costs, which are only implicitly defined as the optimal value of a linear program whose coefficients depend linearly on the uncertain problem parameters [46, Section 2.1]. The following corollary shows how we can evaluate the worst-case expectation of the recourse costs with respect to an ambiguous parameter distribution that is only observable through a finite training dataset.For ease of notation and without loss of generality, we suppress here any dependence on the first-stage decisions.

Corollary 5.4 (Two-stage stochastic programming) Suppose that the uncertainty set is a polytope of the form $\Xi = \{ \xi \in \mathbb { R } ^ { m } : C \xi \leq d \}$ as in Corollaries 5.1 and 5.3.

(i）If $\ell ( \xi ) = \operatorname* { i n f } _ { y } \left\{ \left. y , Q \xi \right. : W y \geq h \right\}$ is the optimal value of a parametric linear program with objective uncertainty, and if the feasible set $\{ y \ : \ W y \ \geq \ h \}$ is non-empty and compact, then the worst-case expectation (1O) is given by

$$
\left\{ \begin{array} { l l l } { ~ \displaystyle \operatorname* { i n f } _ { \lambda , s _ { i } , \gamma _ { i } , y _ { i } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \quad \mathrm { s . t . } } & { \left. y _ { i } , Q \widehat { \xi } _ { i } \right. + \left. \gamma _ { i } , d - C \widehat { \xi } _ { i } \right. \leq s _ { i } } & { \forall i \leq N } \\ { \quad } & { W y _ { i } \geq h } & { \forall i \leq N } \\ { \quad } & { \| Q ^ { \top } y _ { i } - C ^ { \top } \gamma _ { i } \| _ { * } \leq \lambda } & { \forall i \leq N } \\ { \quad } & { \gamma _ { i } \geq 0 } & { \forall i \leq N . } \end{array} \right.
$$

(i） If $\ell ( \xi ) = \operatorname* { i n f } _ { y } \left\{ \left. q , y \right. : W y \geq H \xi + h \right\}$ is the optimal value of a parametric linear program with right-hand side uncertainty, and if the dual feasible set $\{ \theta \geq 0 : W ^ { \intercal } \theta = q \}$ is non-empty and compact with vertices $v _ { k }$ ， $k \leq K$ , then the worst-case expectation (1O) is given by

$$
\left\{ \begin{array} { l l } { \displaystyle { \operatorname* { i n f } _ { \lambda , s _ { i } , \gamma _ { i k } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } } \\ { \mathrm { ~ s . t . ~ } \left. v _ { k } , h \right. + \left. H ^ { \intercal } v _ { k } , \widehat { \xi _ { i } } \right. + \left. \gamma _ { i k } , d - C \widehat { \xi _ { i } } \right. \leq s _ { i } } & { \forall i \leq N , \forall k \leq K } \\ { \| C ^ { \intercal } \gamma _ { i k } - H ^ { \intercal } v _ { k } \| _ { * } \leq \lambda } & { \forall i \leq N , \forall k \leq K } \\ { \gamma _ { i k } \geq 0 } & { \forall i \leq N , \forall k \leq K . } \end{array} \right.
$$

Proof Assertion (i) follows directly from Theorem 4.2 because $\ell ( \xi )$ is concave as an infimum of linear functions in $\xi$ . Indeed, the compactness of the feasible set $\{ y :$ $W y \geq h \}$ ensures that Assumption 4.1 holds for $K = 1$ . In this setting, we find

$$
{ \begin{array} { r l } & { [ - \ell ] ^ { * } ( z ) = \operatorname* { s u p } _ { \xi } \left\{ \langle z , \xi \rangle + \operatorname* { i n f } _ { y } \left\{ \langle y , Q \xi \rangle : W y \geq h \right\} \right\} } \\ & { \qquad = \operatorname* { i n f } _ { y } \left\{ \operatorname* { s u p } _ { \xi } \left\{ \langle z + Q ^ { \mathsf { T } } y , \xi \rangle \right\} : W y \geq h \right\} } \\ & { \qquad = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ t h e r e ~ e x i s t s ~ } } y { \mathrm { ~ w i t h ~ } } Q ^ { \mathsf { T } } y = - z { \mathrm { ~ a ~ } } } \\ { \infty } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. } } \end{array} }
$$

where the second equality follows from the classical minimax theorem [4,Proposition 5.5.4], which applies because $\{ y : W y \geq h \}$ is compact. Assertion (i) then follows by substituting $[ - \ell ] ^ { * }$ as well as the formula for $\sigma _ { \Xi }$ from Corollary 5.1 into (11).

Assertion (ii) relies on the following reformulation of the loss function,

$$
\begin{array} { l } { \ell ( \xi ) = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { y } \left. q , y \right. } \\ { \mathrm { s . t . } W y \geq H \xi + h } \end{array} \right. = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { s u p } _ { \theta \geq 0 } \left. \theta , H \xi + h \right. } \\ { \displaystyle \mathrm { s . t . } W ^ { \intercal } \theta = q } \end{array} \right. = \operatorname* { m a x } _ { k \leq K } \left. v _ { k } , H \xi + h \right. } \\ { \displaystyle = \operatorname* { m a x } _ { k \leq K } \left. H ^ { \intercal } v _ { k } , \xi \right. + \left. v _ { k } , h \right. , } \end{array}
$$

where the first equality holds due to strong linear programming duality, which applies as the dual feasible set is non-empty. The second equality exploits the elementary observation that the optimal value of a linear program with non-empty, compact feasible set is always adopted at a vertex. As we managed to express $\ell ( \xi )$ as a pointwise maximum of linear functions, assertion (ii) follows immediately from Corllary 5.1 (i).

As expected, in the ambiguity-free limit, problem (18a) reduces to a standard SAA problem. Indeed, for $\varepsilon = 0$ , the variable $\lambda$ can be made large at no penalty, and thus $\gamma _ { i } = 0$ and $s _ { i } = \left. y _ { i } , Q \widehat { \xi _ { i } } \right.$ at optimality. In this case, problem (18a) is equivalent to

$$
\operatorname* { i n f } _ { y _ { i } } \left\{ { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \left. y _ { i } , { \mathcal { Q } } { \widehat { \xi } } _ { i } \right. : W y _ { i } \geq h \forall i \leq N \right\} .
$$

Similarly, one can verify that for $\varepsilon = 0$ ,(18b) reduces to the SAA problem

$$
\operatorname* { i n f } _ { y _ { i } } \left\{ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. y _ { i } , q \right. : W y _ { i } \geq H \widehat { \xi } _ { i } \forall i \leq N \right\} .
$$

We close this section with a remark on the computational complexity of all the convex optimization problems derived in this section.

# Remark 5.5 (Computational tractability)

·Ifthe Wasserstein metricis defined interms of the1-norm (i.e. $\begin{array} { r } { \| \xi \| = \sum _ { k = 1 } ^ { m } | \xi _ { k } | ) } \end{array}$ or the $\infty$ -norm (i.e., $\left\| \boldsymbol { \xi } \right\| = \operatorname* { m a x } _ { k \leq m } \left| \xi _ { k } \right| )$ , then the optimization problems (15a), (15b),(17a), (17b), (18a) and (18b) all reduce to linear programs whose sizes scale with the number $N$ of data points and the number $J$ of affine pieces of the underlying loss functions.   
Except for the two-stage stochastic program with right-hand side uncertainty in (18b), the resulting linear programs scale polynomially in the problem description and are therefore computationally tractable.As the number of vertices $v _ { k } , k \le K$ ， of the polytope $\{ \theta \geq 0 : W ^ { \intercal } \theta = q \}$ may be exponential in the number of its facets, however, the linear program (18b) has generically exponential size.   
Inspecting (15a),one easily verifies that the distributionally robust optimization problem(5） reduces to a finite convex program if $\mathbb { X }$ is convex and $h ( x , \xi ) =$ $\begin{array} { r } { \operatorname* { m a x } _ { k \le K } \left. a _ { k } ( x ) , { \xi } \right. + b _ { k } ( x ) } \end{array}$ ， while the gradients $a _ { k } ( x )$ and the intercepts $b _ { k } ( x )$ depend linearly on $x$ . Similarly,(5) can be reformulated as a finite convex program if $\mathbb { X }$ is convex and $h ( x , \xi ) \ : = \ : \operatorname* { i n f } _ { y } \left\{ \left. y , Q \xi \right. : W y \geq h ( x ) \right\}$ or $h ( x , \xi ) =$ $\operatorname { i n f } _ { y }$ $\left\{ \left. q , y \right. : W y \geq H ( x ) \xi + h ( x ) \right\}$ ，while the right hand side coefficients $h ( x )$ and $H ( x )$ depend linearly on $x$ ； see (18a) and (18b), respectively. In contrast, problems (15b),(17a) and (17b) result in non-convex optimization problems when their data depends on $x$   
· We emphasize that the computational complexity of all convex programs examined in this section is independent of the radius $\varepsilon$ of the Wasserstein ball.

# 6 Tractable extensions

We now demonstrate that through minor modifications of the proofs, Theorems 4.2 and 4.4 extend to worst-case expectation problems involving even richer classes of loss functions. First, we investigate problems where the uncertainty can be viewed as a stochastic process and where the loss function is additively separable.Next, we study problems whose loss functions are convex in the uncertain variables and are therefore not necessarily representable as finite maxima of concave functions as postulated by Assumption 4.1.

# 6.1 Stochastic processes with a separable cost

Consider a variant of the worst-case expectation problem (1O), where the uncertain parameters can be interpreted as a stochastic process $\xi = ( \xi _ { 1 } , \dots , \xi _ { T } )$ , and assume that $\xi _ { t } \in \Xi _ { t }$ ，where $\Xi _ { t } \subset \mathbb { R } ^ { m }$ is non-empty and closed for any $t \leq T$ . Moreover, assume that the loss function is additively separable with respect to the temporal structure of $\xi$ , that is,

$$
\ell ( \boldsymbol { \xi } ) { : = } \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { k \leq K } \ell _ { t k } \big ( \boldsymbol { \xi } _ { t } \big ) ,
$$

where $\ell _ { t k } : \mathbb { R } ^ { m } \to \overline { { \mathbb { R } } }$ is a measurable function for any $k \leq K$ and $t \leq T$ . Such loss functions appear, for instance,in open-loop stochastic optimal control or in multi-item newsvendorproblems.Consideraprocessom $\begin{array} { r } { \| \xi \| _ { \mathrm { T } } = \sum _ { t = 1 } ^ { T } \| \xi _ { t } \| } \end{array}$ assciated with the base norm $\| \cdot \|$ on $\mathbb { R } ^ { m }$ , and assume that its induced metric is the one used in the definition of the Wasserstein distance. Note that if $\| \cdot \|$ is the 1-norm on $\mathbb { R } ^ { m }$ , then $\lVert \cdot \rVert _ { \mathrm { T } }$ reduces to the 1-norm on RmT.

By interchanging summation and maximization, the loss function (19) can be reexpressed as

$$
\ell ( \boldsymbol { \xi } ) = \operatorname* { m a x } _ { k _ { t } \leq K } \sum _ { t = 1 } ^ { T } \ell _ { t k _ { t } } \left( \xi _ { t } \right) ,
$$

where the maximum runs over all $K ^ { T }$ combinations of $k _ { 1 } , \dots , k _ { T } \le K$ . Under this representation, Theorem 4.2 remains applicable. However, the resulting convex optimization problem would involve $\mathcal { O } ( K ^ { T } )$ decision variables and constraints, indicating that an efcient solution may not be available.Fortunately, this deficiency can be overcome by modifying Theorem 4.2.

Theorem 6.1 (Convex reduction for separable loss functions） Assume that the loss function $\ell$ is of the form (19), and the Wasserstein ball is defined through the process norm $\lVert \cdot \rVert _ { \mathrm { T } }$ . Then, for any $\varepsilon \geq 0$ ，the worst-case expectation (1O) is smaller or equal to the optimal value of the finite convex program

$$
\left\{ \begin{array} { l l } { \displaystyle { \operatorname* { i n f } _ { \boldsymbol { \lambda } , \boldsymbol { \varepsilon } _ { t i } , \boldsymbol { \varepsilon } _ { t i } , \boldsymbol { \nu } _ { t i k } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } s _ { t i } } } \\ { \displaystyle { \mathrm { s . t . } \mathrm { s . t . } \mathrm { ~ } [ - \ell _ { t k } ] ^ { * } \big ( z _ { t i k } - \nu _ { t i k } \big ) + \sigma _ { \Xi _ { t } } ( \nu _ { t i k } ) - \big \langle z _ { t i k } , \widehat { \xi } _ { t i } \big \rangle \leq s _ { t i } } } & { \forall i \leq N , \forall k \leq K , \forall t \leq T , } \\ { \| z _ { t i k } \| _ { * } \leq \lambda } & { \forall i \leq N , \forall k \leq K , \forall t \leq T . } \end{array} \right.
$$

f $\Xi _ { t }$ and $\{ \ell _ { t k } \} _ { k \le K }$ satisfy the convexity Assumption 4.1 for every $t \leq T$ ， then the worst-case expectation (1O) coincides exactly with the optimal value of problem (20).

Proof Up until equation (12d), the proof of Theorem 6.1 parallels that of Theorem 4.2. Starting from (12d), we then have

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ] = \operatorname* { i n f } _ { \lambda \geq 0 } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { s u p } _ { \xi } \big ( \ell ( \xi ) - \lambda \| \xi - \widehat { \xi } _ { i } \| _ { \mathrm { T } } \big ) } } \\ & { } & { = \operatorname* { i n f } _ { \lambda \geq 0 } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \operatorname* { s u p } _ { \xi _ { t } \in \Xi _ { t } } \bigg ( \operatorname* { m a x } _ { k \leq K } \ell _ { t k } \big ( \xi _ { t } \big ) - \lambda \| \xi _ { t } - \widehat { \xi } _ { t i } \| \bigg ) , } \end{array}
$$

where the interchange of the summation and the maximization is facilitated by the separability of the overall loss function. Introducing epigraphical auxiliary variables yields

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { \lambda , s _ { t i } } \ \lambda s + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } s _ { t i } } \\ { \mathrm { s . t . } \ \displaystyle \operatorname* { s u p } _ { \xi _ { t } \in \Xi _ { t } } \left( \ell _ { t k } \left( \xi _ { t } \right) - \lambda \left\| \xi _ { t } - \widehat { \xi } _ { t i } \right\| \right) \leq s _ { t i } \quad \forall i \leq N , \ \forall k \leq K , \ \forall t \leq T } \\ { \quad \quad \quad \lambda \geq 0 } \end{array} \right.
$$

$$
\begin{array} { r l } & { \le \left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { \lambda , s t , z \neq i \lambda } \lambda s + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } s _ { t i } } \\ { \displaystyle s . t . } \\ { \displaystyle s . t . } \\ { \displaystyle \quad \xi _ { i } \in \mathbb { E } _ { t } } \left( \ell _ { k } \left( \xi _ { i } \right) - \left. z _ { i i k } , \xi _ { i } \right. \right) + \left. z _ { i i k } , \widehat { \xi } _ { t i } \right. \le s _ { t i } \right.} & { \forall i \le N , \forall k \le K , \forall t \le 7 } \\ { \displaystyle \qquad \left. z _ { i k } \right. _ { \ast \ast } \| z _ { i k } \| _ { * } \le \lambda } & { \forall i \le N , \forall k \le K , \forall t \le 7 } \end{array}   \\ & { = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { \lambda , s t , z \neq i \lambda } \frac { N } { \lambda } \sum _ { i = 1 } ^ { N } s _ { t i } } \\ { \displaystyle s . t . } \\ { \displaystyle \operatorname* { s t } _ { \lambda , t , \quad } \left[ - \ell _ { t k } + \chi _ { z i } \right] ^ { \ast } \left( - z _ { t i k } \right) + \left. z _ { t i k } , \widehat { \xi } _ { t i } \right. \le s _ { t i } } & { \forall i \le N , \forall k \le K , \forall t \le T } \\ { \displaystyle \forall i \le N , \forall k \le K , \forall t \le T , } \end{array} \right. } \end{array}
$$

where the inequality is justified in a similar manner as the one in (12e),and it holds as an equality provided that $\Xi _ { t }$ and $\{ \ell _ { t k } \} _ { k \le K }$ satisfy Assumption 4.1 for all $t \leq T$ Finally, by Rockafellar and Wets [42, Theorem 11.23(a),p. 493], the conjugate of $- \ell _ { t k } + \chi _ { \Xi _ { t } }$ can be replaced by the inf-convolution of the conjugates of $- \ell _ { t k }$ and $\mathbb { X } \Xi _ { t }$ This completes the proof. □

Note that the convex program (2O) involves only $\mathcal { O } ( N K T )$ decision variables and constraints. Moreover, if $\ell _ { t k }$ is affine for every $t \leq T$ and $k \leq K$ , while $\| \cdot \|$ represents the 1-norm or the $\infty$ -norm on $\mathbb { R } ^ { m }$ , then (2O) reduces to a tractable linear program (see also Remark 5.5). A natural generalization of Theorem 4.4 further allows us to characterize the extremal distributions of the worst-case expectation problem (1O) with a separable loss function of the form (19).

Theorem 6.2 (Worst-case distributions for separable loss functions） Assume that the loss function $\ell$ is of the form (19), and the Wasserstein ball is defined through the process norm $\lVert \cdot \rVert _ { \mathrm { T } }$ If $\Xi _ { t }$ and $\{ \ell _ { t k } \} _ { k \le K }$ satisfy Assumption 4.1 for all $t \ \leq \ T$ ，then the worst-case expectation (1O) coincides with the optimal value of the finite convex program

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { s u p } _ { \alpha _ { i k } , g _ { i i k } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { T } \sum _ { t = 1 } ^ { T } \alpha _ { t i k } \ell _ { t k } \Bigl ( \widehat { \xi } _ { t i } - \frac { q _ { t i k } } { \alpha _ { t i k } } \Bigr ) } & \\ { \mathrm { s . t . } \quad \displaystyle _ { \mathrm { N } } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \sum _ { t = 1 } ^ { T } \| q _ { t i k } \| \le \varepsilon } \\ { \quad \ } & { \displaystyle \sum _ { k = 1 } ^ { K } \alpha _ { t i k } = 1 } & { \forall i \le N , \forall t \le T } \\ { \quad \ } & { \displaystyle \alpha _ { t i k } \ge 0 } \\ { \quad \quad } & { \displaystyle \forall i \le N , \forall t \le T , \forall k \le K } \\ { \quad \quad \widetilde { \xi } _ { t i } - \frac { q _ { t i k } } { \alpha _ { t i k } } \in \Xi _ { t } } & { \forall i \le N , \forall t \le T , \forall k \le K } \end{array} \right.
$$

irespective of $\varepsilon \geq 0$ Let $\left\{ \alpha _ { t i k } ( r ) , q _ { t i k } ( r ) \right\} _ { r \in \mathbb { N } }$ be a sequence of feasible decisions whose objective values converge to the supremum of (21). Then, the discrete (product) probability distributions

$$
\mathbb { Q } _ { r } { : = } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigotimes _ { t = 1 } ^ { T } \Big ( \sum _ { k = 1 } ^ { K } \alpha _ { t i k } ( r ) \delta _ { \xi _ { t i k } ( r ) } \Big ) \quad w i t h \quad \xi _ { t i k } ( r ) { : = } \widehat { \xi } _ { t i } - \frac { q _ { t i k } ( r ) } { \alpha _ { t i k } ( r ) }
$$

belong to the Wasserstein ball $\mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ and attain the supremum of (10)asymptotically, i.e.，

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ] = \operatorname* { l i m } _ { r  \infty } \mathbb { E } ^ { \mathbb { Q } _ { r } } \big [ \ell ( \xi ) \big ] = \operatorname* { l i m } _ { r  \infty } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \sum _ { t = 1 } ^ { T } \alpha _ { t i k } ( r ) \ell _ { t k } \big ( \xi _ { t i k } ( r ) \big ) .
$$

Proof As in the proof of Theorem 4.4, the claim follows by dualizing the convex program (2O). Details are omitted for brevity of exposition. □

We emphasize that the distributions $\mathbb { Q } _ { r }$ from Theorem 6.2 can be constructed eficiently by solving a convex program of polynomial size even though they have $N K ^ { T }$ discretization points.

# 6.2 Convex loss functions

Consider now another variant of the worst-case expectation problem (1O), where the loss function $\ell$ is proper, convex and lower semicontinuous.Unless $\ell$ is piecewise affine, we cannot represent such a loss function as a pointwise maximum of finitely many concave functions,and thus Theorem 4.2 may only provide a loose upper bound on the worst-case expectation (1O). The following theorem provides an alternative upper bound that admits new insights into distributionally robust optimization with Wasserstein balls and becomes exact for $\Xi = \mathbb { R } ^ { m }$

Theorem 6.3 (Convex reduction for convex loss functions） Assume that the loss function $\ell$ is proper, convex, and lower semicontinuous, and define $\kappa { : = } \operatorname* { s u p } \left\{ \| \theta \| _ { * } \right. :$ $\ell ^ { * } ( \theta ) < \infty \}$ . Then, for any $\varepsilon \geq 0$ , the worst-case expectation (10)is smaller or equal to

$$
\kappa \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( \widehat { \xi } _ { i } ) .
$$

f $\Xi = \mathbb { R } ^ { m }$ , then the worst-case expectation (1O) coincides exactly with (22).

Remark 6.4 (Radius of effective domain) The parameter $\kappa$ can be viewed as the radius of the smallest ball containing the effective domain of the conjugate function $\ell ^ { * }$ in terms of the dual norm. By the standard conventions of extended arithmetic, the term $\kappa \varepsilon$ in (22) is interpreted as O if $\kappa = \infty$ and $\varepsilon = 0$

Proof Equation (12b) in the proof of Theorem 4.2 implies that

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \big [ \ell ( \xi ) \big ] = \operatorname* { i n f } _ { \lambda \geq 0 } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { s u p } _ { \xi \in \Xi } \big ( \ell ( \xi ) - \lambda \| \xi - \widehat { \xi } _ { i } \| \big )
$$

for every $\varepsilon > 0$ As $\ell$ is proper, convex, and lower semicontinuous, it coincides with its bi-conjugate function $\ell ^ { * * }$ , see e.g. [4, Proposition 1.6.1(c)]. Thus, we may write

$$
\ell ( \xi ) = \operatorname* { s u p } _ { \theta \in \Theta } \left. \theta , \xi \right. - \ell ^ { \ast } ( \theta ) ,
$$

where $\Theta \mathrm { : = } \{ \theta \in \mathbb { R } ^ { m } : \ell ^ { \ast } ( \theta ) < \infty \}$ denotes the effective domain of the conjugate function $\ell ^ { * }$ . Using this dual representation of $\ell$ in conjunction with the definition of the dual norm, we find

$$
\begin{array} { r l } { \underset { \xi \in \Xi } { \operatorname* { s u p } } \left( \ell ( \xi ) - \lambda \lVert \xi - \widehat { \xi } _ { i } \rVert \right) = \underset { \xi \in \Xi } { \operatorname* { s u p } } } & { \underset { \theta \in \Theta } { \operatorname* { s u p } } \left( \left. \theta , \xi \right. - \ell ^ { * } ( \theta ) - \lambda \lVert \xi - \widehat { \xi } _ { i } \rVert \right) } \\ { = } & { \underset { \xi \in \Xi } { \operatorname* { s u p } } } & { \underset { \theta \in \Theta } { \operatorname* { s u p } } \underset { \| z \| _ { * } \leq \lambda } { \operatorname* { i n f } } \left( \left. \theta , \xi \right. - \ell ^ { * } ( \theta ) + \left. z , \xi \right. - \left. z , \widehat { \xi } _ { i } \right. \right) . } \end{array}
$$

The classical minimax theorem [4,Proposition 5.5.4] then allows us to interchange the maximization over $\xi$ with the maximization over $\theta$ and the minimization over $z$ to obtain

$$
\begin{array} { r l } & { \underset { \xi \in \Xi } { \operatorname* { s u p } } \left( \ell ( \xi ) - \lambda \lVert \xi - \widehat { \xi } _ { i } \rVert \right) = \underset { \theta \in \Theta } { \operatorname* { s u p } } \ \underset { \| z \| _ { * } \leq \lambda } { \operatorname* { i n f } } \ \underset { \xi \in \Xi } { \operatorname* { s u p } } \left( \left. \theta + z , \xi \right. - \ell ^ { * } ( \theta ) - \left. z , \widehat { \xi } _ { i } \right. \right) } \\ & { \qquad = \underset { \theta \in \Theta } { \operatorname* { s u p } } \ \underset { \| z \| _ { * } \leq \lambda } { \operatorname* { i n f } } \ \sigma _ { \Xi } ( \theta + z ) - \ell ^ { * } ( \theta ) - \left. z , \widehat { \xi } _ { i } \right. . } \end{array}
$$

Recall that $\sigma _ { \Xi }$ denotes the support function of $\Xi$ . It seems that there is no simple exact reformulation of (24） for arbitrary convex uncertainty sets $\Xi$ . Interchanging the maximization over $\theta$ with the minimization over $z$ in （24） would lead to the conservative upper bound of Corollary 4.3. Here,however, we employ an alternative approximation. By definition of the support function, we have $\sigma _ { \Xi } \leq \sigma _ { \mathbb { R } ^ { m } } = \chi _ { \{ 0 \} }$ Replacing $\sigma _ { \Xi }$ with $\chi _ { \{ 0 \} }$ in (24) thus results in the conservative approximation

$$
\operatorname* { s u p } _ { \xi \in \mathbb { R } ^ { m } } \Big ( \ell ( \xi ) - \lambda \| \xi - \widehat { \xi } _ { i } \| \Big ) \leq \left\{ \begin{array} { l l } { \ell ( \widehat { \xi } _ { i } ) } & { \mathrm { i f ~ } \operatorname* { s u p } \big \{ \| \theta \| _ { * } : \theta \in \Theta \big \} \leq \lambda , } \\ { \infty } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The inequality (22) then follows readily by substituting(25) into (23) and using the definition of $\kappa$ in the theorem statement. For $\Xi = \mathbb { R } ^ { m }$ we have $\sigma _ { \Xi } = \chi _ { \{ 0 \} }$ , and thus the upper bound (22) becomes exact. Finally, if $\varepsilon = 0$ , then（1O） trivially coincides with (22) under our conventions of extended arithmetic. Thus, the claim follows.□

Theorem 6.3 asserts that for $\Xi = \mathbb { R } ^ { m }$ , the worst-case expectation (10) of a convex loss function reduces the sample average of the loss adjusted by the simple correction term $\kappa \varepsilon$ . The following proposition highlights that $\kappa$ can be interpreted as a measure of maximum steepness of the loss function. This interpretation has intuitive appeal in view of Definition 3.1.

Proposition 6.5 (Steepness of the loss function) Let k be defined as in Theorem 6.3.

(i) Iflis $\overline { { L } }$ -Lipschitz continuous, i.e.,ifthere exists $\boldsymbol { \xi } ^ { \prime } \in \mathbb { R } ^ { m }$ such that $\ell ( \xi ) - \ell ( \xi ^ { \prime } ) \leq$ ${ \overline { { L } } } \| \xi - \xi ^ { \prime } \|$ for all $\xi \in \mathbb { R } ^ { m }$ ,then $\kappa \leq \overline { { L } }$   
(ii） If $\ell$ majorizes an affine function, i.e., if there exists $\theta \in \mathbb { R } ^ { m }$ with $\| \theta \| _ { * } = : \underline { { L } }$ and $\boldsymbol { \xi } ^ { \prime } \in \mathbb { R } ^ { m }$ such that $\ell ( \xi ) - \ell ( \xi ^ { \prime } ) \geq \left. \theta , \xi - \xi ^ { \prime } \right.$ for all $\xi \in \mathbb { R } ^ { m }$ ,then $\kappa \geq \underline { { L } }$

Proof The proof follows directly from the definition of conjugacy. As for (i), we have

$$
\begin{array} { r l } { \ell ^ { * } ( \theta ) = \displaystyle \operatorname* { s u p } _ { \xi \in \mathbb { R } ^ { m } } \left. \theta , \xi \right. - \ell ( \xi ) \geq \displaystyle \operatorname* { s u p } _ { \xi \in \mathbb { R } ^ { m } } \left. \theta , \xi \right. - \overline { { L } } \| \xi - \xi ^ { \prime } \| - \ell ( \xi ^ { \prime } ) } & { } \\ { = \displaystyle \operatorname* { s u p } _ { \xi \in \mathbb { R } ^ { m } } \operatorname* { i n f } _ { \| z \| _ { \xi } \leq \overline { { L } } } \left. \theta , \xi \right. - \left. z , \xi - \xi ^ { \prime } \right. - \ell ( \xi ^ { \prime } ) , } & { } \end{array}
$$

where the last equality follows from the definition of the dual norm. Applying the minimax theorem [4, Proposition 5.5.4] and explicitly carrying out the maximization over $\xi$ yields

$$
\ell ^ { * } ( \theta ) \geq \left\{ \begin{array} { l l } { \left. \theta , \xi ^ { \prime } \right. - \ell ( \xi ^ { \prime } ) } & { \mathrm { i f } \ \| \theta \| _ { * } \leq \overline { { L } } , } \\ { \infty } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Consequently, $\ell ^ { * } ( \theta )$ is infinite for all $\theta$ with $\lVert \theta \rVert _ { * } > \overline { { L } }$ , which readily implies that the $\| \cdot \| _ { * }$ -ball of radius $\overline { { L } }$ contains the effective domain of $\ell ^ { * }$ . Thus, $\kappa \leq \overline { { L } }$

As for (ii), we have

$$
\begin{array} { r l } & { \ell ^ { * } ( \theta ) = \underset { \xi \in \mathbb { R } ^ { m } } { \operatorname* { s u p } } \left. \theta , \xi \right. - \ell ( \xi ) \leq \underset { \xi \in \mathbb { R } ^ { m } } { \operatorname* { s u p } } \left. \theta , \xi \right. - \left. z , \xi - \xi ^ { \prime } \right. - \ell ( \xi ^ { \prime } ) } \\ & { \qquad = \sigma _ { \mathbb { R } ^ { m } } ( \theta - z ) + \left. z , \xi ^ { \prime } \right. - \ell ( \xi ^ { \prime } ) , } \end{array}
$$

which implies that $\ell ^ { * } ( \theta ) \leq \left. \theta , \xi ^ { \prime } \right. - \ell ( \xi ^ { \prime } ) < \infty$ Thus, $\theta$ belongs to the effective domain of $\ell ^ { * }$ . We then conclude that $\kappa \geq \| \theta \| _ { * } = \underline { { L } }$ □

Remark 6.6 (Consistent formulations） If $\Xi = \mathbb { R } ^ { m }$ and the loss function is given by $\ell ( \xi ) = \mathrm { m a x } _ { k \le K } \{ \left. a _ { k } , \xi \right. + b _ { k } \}$ , then both Corollary 5.1 and Theorem 6.3 offer an exact reformulation of the worst-case expectation (1O) in terms of a finite-dimensional convex program. On the one hand, Corollary 5.1 implies that (1O) is equivalent to

$$
\left\{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } _ { \lambda } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( \widehat \xi _ { i } ) } \\ { \mathrm { s . t . } \ \| a _ { k } \| _ { * } \leq \lambda } & { \forall k \leq K , } \end{array} \right.
$$

which is obtained by setting $C \ = \ 0$ and $d \ : = \ : 0$ in (15a). At optimality we have $\lambda ^ { \star } = \operatorname* { m a x } _ { k \leq K } \| a _ { k } \| _ { * }$ , which corresponds to the (best) Lipschitz constant of $\ell ( \xi )$ with respect to the norm $\| \cdot \|$ . On the other hand, Theorem 6.3 implies that (10) is equivalent to (22) with $\kappa = \lambda ^ { \star }$ . Thus, Corollary 5.1 and Theorem 6.3 are consistent.

Remark 6.7 ( $\varepsilon$ -insensitive optimizers3） Consider a loss function $h ( x , \xi )$ that is convex in $\xi$ , and assume that $\Xi = \mathbb { R } ^ { m }$ . In this case Theorem 6.3 remains valid, but the steepness parameter $\kappa ( x )$ may depend on $x$ . For loss functions whose Lipschitz modulus with respect to $\xi$ is independent of $x$ (e.g., the newsvendor loss), however, $\kappa ( x )$ is constant. In this case the distributionally robust optimization problem (5) and the SAA problem (4) share the same minimizers irrespective of the Wasserstein radius $\varepsilon$ This phenomenon could explain why the SAA solutions tend to display a surprisingly strong out-of-sample performance in these problems.

# 7 Numerical results

We validate the theoretical results of this paper in the context of a stylized portfolio selection problem. The subsequent simulation experiments are designed to provide additional insights into the performance guarantees of the proposed distributionally robust optimization scheme.

# 7.1 Mean-risk portfolio optimization

Consider a capital market consisting of $m$ assets whose yearly returns are captured   
by the random vector $\boldsymbol { \xi } = [ \xi _ { 1 } , \ldots , \xi _ { m } ] ^ { \intercal }$ . If short-selling is forbidden,a portfolio $x = [ x _ { 1 } , \ldots , x _ { m } ] ^ { \intercal }$ $\mathbb { X } = \{ x \in \mathbb { R } _ { + } ^ { m } : \sum _ { i = 1 } ^ { m } x _ { i } = 1 \}$ $x$   
$x _ { i }$ of the available capital in asset $i$ for each $i = 1 , \ldots , m$ , its return amounts to $\left. { x , \xi } \right.$   
In the remainder we aim to solve the single-stage stochastic program

$$
J ^ { \star } = \operatorname* { i n f } _ { x \in \mathbb { X } } \bigg \{ \mathbb { E } ^ { \mathbb { P } } \big [ - \langle x , \xi \rangle \big ] + \rho \mathbb { P } \mathrm { - C V a R } _ { \alpha } \big ( - \langle x , \xi \rangle \big ) \bigg \} ,
$$

which minimizes a weighted sum of the mean and the conditional value-at-risk(CVaR) of the portfolio loss $- \langle x , \xi \rangle$ ,where $\alpha \in ( 0 , 1 ]$ is referred to as the confidence level of the CVaR, and $\rho \in \mathbb { R } _ { + }$ quantifies the investor's risk-aversion. Intuitively, the CVaR at level $\alpha$ represents the average of the $\alpha \times 1 0 0 \%$ worst (highest） portfolio losses under the distribution P.Replacing the CVaR in the above expression with its formal definition [41], we obtain

$$
\begin{array} { r l } & { J ^ { \star } = \underset { x \in \mathbb { X } } { \operatorname* { i n f } } \Big \{ \mathbb { E } ^ { \mathbb { P } } \big [ - \langle x , \xi \rangle \big ] + \rho \underset { \tau \in \mathbb { R } } { \operatorname* { i n f } } \mathbb { E } ^ { \mathbb { P } } \Big [ \tau + \frac { 1 } { \alpha } \operatorname* { m a x } \big \{ - \langle x , \xi \rangle - \tau , 0 \big \} \Big ] \Big \} } \\ & { \quad = \underset { x \in \mathbb { X } , \tau \in \mathbb { R } } { \operatorname* { i n f } } \mathbb { E } ^ { \mathbb { P } } \Big [ \underset { k \leq K } { \operatorname* { m a x } } a _ { k } \big \langle x , \xi \big \rangle + b _ { k } \tau \Big ] , } \end{array}
$$

where $K = 2$ ， $a _ { 1 } = - 1$ ， $\begin{array} { r } { a _ { 2 } = - 1 - \frac { \rho } { \alpha } , b _ { 1 } = \rho } \end{array}$ and $b _ { 2 } = \rho ( 1 - \textstyle { \frac { 1 } { \alpha } } )$ . An investor who is unaware of the distribution $\mathbb { P }$ but has observed a dataset $\widehat { \Xi } _ { N }$ of $N$ historical samples from $\mathbb { P }$ and knows that the support of $\mathbb { P }$ is contained in $\Xi = \{ \xi \in \mathbb { R } ^ { m } : C \xi \leq d \}$ might solve the distributionally robust counterpart of (26) with respect to the Wasserstein ambiguity set $\mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ , that is,

$$
\widehat { J } _ { N } ( \varepsilon ) : = \operatorname* { i n f } _ { \substack { x \in \mathbb { X } , \tau \in \mathbb { R } } } \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { E } ^ { \mathbb { Q } } \Big [ \operatorname* { m a x } _ { k \leq K } a _ { k } \big \langle x , \xi \big \rangle + b _ { k } \tau \Big ] ,
$$

where we make the dependence on the Wasserstein radius $\varepsilon$ explicit. By Corollary 5.1 we know that

$$
\widehat { J } _ { N } ( \varepsilon ) = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { i n f } _ { x , \tau , \lambda , s _ { i } , \gamma _ { i k } } \lambda \varepsilon + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } } \\ { \mathrm { s . t . } \quad x \in \mathbb { X } } \\ { \quad } & { b _ { k } \tau + a _ { k } \big \langle x , \widehat { \xi } _ { i } \big \rangle + \big \langle \gamma _ { i k } , d - C \widehat { \xi } _ { i } \big \rangle \leq s _ { i } \quad \forall i \leq N , \forall k \leq K } \\ { \| C ^ { \top } \gamma _ { i k } - a _ { k } x \| _ { * } \leq \lambda } & { \forall i \leq N , \forall k \leq K } \\ { \quad } & { \gamma _ { i k } \geq 0 \quad \forall i \leq N , \forall k \leq K . } \end{array} \right.
$$

Before proceeding with the numerical analysis of this problem, we provide some analytical insights into its optimal solutions when there is significant ambiguity. In what follows we keep the training data set fixed and let ${ \widehat { x } } _ { N } ( \varepsilon )$ be an optimal distributionally robust portfolio corresponding to the Wasserstein ambiguity set of radius $\varepsilon$ . We will now show that, for natural choices of the ambiguity set, $\widehat { x } _ { N } ( \varepsilon )$ converges to the equally weighted portfolio ${ \frac { 1 } { m } } e$ as $\varepsilon$ tends to infinity, where $e { : = } ( 1 , \ldots , 1 ) ^ { \intercal }$ The optimality of the equally weighted portfolio under high ambiguity has first been demonstrated in [37] using analytical methods. We identify this result here as an immediate consequence of Theorem 4.2, which is primarily a computational result.

For any non-empty set $S \subseteq \mathbb { R } ^ { m }$ we denote by $\operatorname { r e c c } ( S ) { : = } \{ y \ \in \ \mathbb { R } ^ { m } \ : \ x + \lambda y \ \in$ $S \forall x \in S$ ， $\forall \lambda \geq 0 \}$ the recession cone and by $S ^ { \circ } { : = } \{ y \in \mathbb { R } ^ { m } : \left. y , x \right. \leq 0 \forall x \in S \}$ the polar cone of $S$

Lemma 7.1 If $\{ \varepsilon _ { k } \} _ { k \in \mathbb { N } } \subset \mathbb { R } _ { + }$ tends to infinity, then any accumulation point $x ^ { \star }$ of $\left\{ \widehat { x } _ { N } ( \varepsilon _ { k } ) \right\} _ { k \in \mathbb { N } }$ is a portfolio that has minimum distance to $( \mathrm { r e c c } ( \Xi ) ) ^ { \circ }$ with respect to $\| \cdot \| _ { * }$

Proof Note first that $\widehat { x } _ { N } ( \varepsilon _ { k } )$ ， $k \in \mathbb N$ ,and $x ^ { \star }$ exist because $\mathbb { X }$ is compact. For large Wasserstein radii $\varepsilon$ ， the term $\lambda \varepsilon$ dominates the objective function of problem (27). Using standard epi-convergence results [42, Section 7.E], one can thus show that

$$
\begin{array} { r l } & { x ^ { \star } \in \mathop { \arg \operatorname* { m i n } } _ { x \in \mathbb { X } } \underset { \gamma _ { i k } \geq 0 } { \operatorname* { m i n } } _ { \ \ \mathrm { m i } \leq N , k \leq K } \| C ^ { \mathsf { T } } \gamma _ { i k } - a _ { k } x \| _ { * } } \\ & { \quad = \mathop { \arg \operatorname* { m i n } } _ { x \in \mathbb { X } } \underset { i \leq N , k \leq K } { \operatorname* { m a x } } \underset { \gamma \geq 0 } { \operatorname* { m i n } } \ \| C ^ { \mathsf { T } } \gamma + | a _ { k } | x \| _ { * } } \\ & { \quad = \mathop { \arg \underset { x \in \mathbb { X } } { \operatorname* { m i n } } } _ { \gamma \geq 0 } \underset { i \geq 0 } { \operatorname* { m i n } } \ \underset { \gamma \geq 0 } { \operatorname* { m i n } } \ \| C ^ { \mathsf { T } } \gamma + x \| _ { * } \underset { k \leq K } { \operatorname* { m a x } } | a _ { k } | } \\ & { \quad = \mathop { \arg \underset { x \in \mathbb { X } } { \operatorname* { m i n } } } _ { \gamma \geq 0 } \ \underset { \gamma \geq 0 } { \operatorname* { m i n } } \ \| C ^ { \mathsf { T } } \gamma + x \| _ { * } , } \end{array}
$$

where the first equality follows from the fact that $a _ { k } < 0$ for all $k \leq K$ , the second equality uses the substitution $\gamma  \gamma | a _ { k } |$ , and the last equality holds because the set of minimizers of an optimization problem is not affected by a positive scaling of the objective function. Thus, $x ^ { \star }$ is the portfolio nearest to the cone $\mathcal { C } = \{ C ^ { \intercal } \gamma : \gamma \geq 0 \}$ · The claim now follows as the polar cone

$$
\begin{array} { r l } & { \mathcal { C } ^ { \circ } : = \{ y \in \mathbb { R } ^ { m } : y ^ { \intercal } x \leq 0 \forall x \in \mathcal { C } \} = \{ y \in \mathbb { R } ^ { m } : y ^ { \intercal } C ^ { \intercal } \gamma \leq 0 \forall \gamma \geq 0 \} } \\ & { \quad = \{ y \in \mathbb { R } ^ { m } : C y \geq 0 \} } \end{array}
$$

is readily recognized as the recession cone of $\Xi$ and as ${ \mathcal { C } } = ( { \mathcal { C } } ^ { \circ } ) ^ { \circ }$

Proposition 7.2 (Equally weighted portfolio) Assume that the Wasserstein metric is defined in terms of the $p$ -norm in the uncertainty space for some $p \in [ 1 , \infty )$ If $\{ \varepsilon _ { k } \} _ { k \in \mathbb { N } } \subset \mathbb { R } _ { + }$ tends to infinity, then $\left\{ \widehat { x } _ { N } ( \varepsilon _ { k } ) \right\} _ { k \in \mathbb { N } }$ converges to the equally weighted portfolio $\begin{array} { r } { x ^ { \star } = \frac { 1 } { m } e } \end{array}$ provided that the uncertainty set is given by

(i) the entire space,i.e., $\Xi = \mathbb { R } ^ { m }$ ,or

![](images/15bae82f084c54585daf8b9fec17e0743491cececdfc8373746abbd369b458b7.jpg)  
Fig.4 Optimal portfolio composition as a function of the Wasserstein radius $\varepsilon$ averaged over 200 simulations;the portfolio weights are depicted in ascending order,i.e., the weight of asset 1 at the botom (dark blue area)and that of asset 1O at the top (dark red area). (a) $N = 3 0$ training samples. (b) $N = 3 0 0$ training samples. (c) $N = 3 0 0 0$ training samples (color figure online)

(ii) the nonnegative orthant shifted by $- e$ ,i.e., $\Xi = \{ \xi \in \mathbb { R } ^ { m } : \xi \geq - e \}$ ，which captures the idea that no asset can lose more than $100 \%$ of its value.

Proof (i) One easily verifies from the definitions that $( \operatorname { r e c c } ( { \Xi } ) ) ^ { \circ } = \{ 0 \} .$ Moreover, we have $\| \cdot \| _ { * } = \| \cdot \| _ { q }$ where $\begin{array} { r } { \frac { 1 } { p } + \frac { 1 } { q } = 1 } \end{array}$ .As $p \in [ 1 , \infty )$ , we conclude that $q \in ( 1 , \infty ]$ and thus the unique nearest portfolio to $( \mathrm { r e c c } ( \Xi ) ) ^ { \circ }$ with respect to $\| \cdot \| _ { * }$ is $\begin{array} { r } { x ^ { \star } = \frac { 1 } { m } e } \end{array}$ The claim then follows from Lemma 7.1. Assertion (ii) follows in a similar manner from the observation that $( \mathrm { r e c c } ( \Xi ) ) ^ { \circ }$ is now the non-positive orthant. □

With some extra effort one can show that for every $p \in [ 1 , \infty )$ there is a threshold $\bar { \varepsilon } > 0$ with $\widehat { x } _ { N } ( \varepsilon ) = x ^ { \star }$ for all $\varepsilon \geq \overline { \varepsilon }$ ,see [37,Proposition 3]. Moreover, for $p \in \{ 1 , 2 \}$ the threshold $\bar { \varepsilon }$ is known analytically.

# 7.2 Simulation results: portfolio optimization

Our experiments are based on a market with $m = 1 0$ assets considered in [7, Section 7.5]. In view of the capital asset pricing model we may assume that the return $\xi _ { i }$ is decomposable into a systematic risk factor $\psi \sim { \mathcal { N } } ( 0 , 2 \% )$ common to all assets and an unsystematic or idiosyncratic risk factor $\zeta _ { i } \sim \mathcal { N } ( i \times 3 \% , i \times 2 . 5 \% )$ specific to asset $i$ . Thus,we set $\xi _ { i } ~ = ~ \psi + \zeta _ { i }$ ，where $\psi$ and the idiosyncratic risk factors $\zeta _ { i } , i = 1 , \ldots , m$ , constitute independent normal random variables.By construction, assets with higher indices promise higher mean returns at a higher risk.Note that the given moments of the risk factors completely determine the distribution $\mathbb { P }$ of $\xi$ . This distribution has support $\Xi = \mathbb { R } ^ { m }$ and satisfies Assumption 3.3 for the tail exponent $a = 1$ , say. We also set $\alpha = 2 0 \%$ and $\rho = 1 0$ in all numerical experiments, and we use the 1-norm to measure distances in the uncertainty space. Thus, $\| \cdot \| _ { * }$ is the $\infty$ -norm, whereby (27) reduces to a linear program.

# 7.2.1 Impact of the Wasserstein radius

In the first experiment we investigate the impact of the Wasserstein radius $\varepsilon$ on the optimal distributionally robust portfolios and their out-of-sample performance. We solve problem (27） using training datasets of cardinality $N \in \{ 3 0 , 3 0 0 , 3 0 0 0 \}$ .Figure 4 visualizes the corresponding optimal portfolio weights $\widehat { x } _ { N } ( \varepsilon )$ as a function of $\varepsilon$ ,averaged over 2OO independent simulation runs. Our numerical results confirm the theoretical insight of Proposition 7.2 that the optimal distributionally robust portfolios converge to the equally weighted portfolio as the Wasserstein radius $\varepsilon$ increases; see also [37].

![](images/d0b2ba2415466bc772aba004b8c94ed044b1ecc4d04755826137f53e5f8fb3fc.jpg)  
Fig. 5 Out-of-sample performance $J ( \widehat { x } _ { N } ( \varepsilon ) )$ (left axis,solid line and shaded area) and reliability $\mathbb { P } ^ { \overset {  } { N } } [ J ( \widehat { x } _ { N } ( \varepsilon ) ) \ \leq \ \widehat { J } _ { N } ( \varepsilon ) ]$ (right axis, dashed line) as a function of the Wasserstein radius $\varepsilon$ and estimated on the basis of 200 simulations. (a) $N = 3 0$ training samples. (b) $N = 3 0 0$ training samples. (c) $N = 3 0 0 0$ training samples

The out-of-sample performance

$$
J \left( \widehat { x } _ { N } ( \varepsilon ) \right) : = \mathbb { E } ^ { \mathbb { P } } \big [ - \left. \widehat { x } _ { N } ( \varepsilon ) , \xi \right. \big ] + \rho \mathbb { P } \mathrm { - } \mathrm { C V a R } _ { \alpha } \big ( - \left. \widehat { x } _ { N } ( \varepsilon ) , \xi \right. \big )
$$

of any fixed distributionally robust portfolio $\widehat { x } _ { N } ( \varepsilon )$ can be computed analytically as $\mathbb { P }$ constitutes a normal distribution by design, see, e.g., [41, p. 29]. Figure 5 shows the tubes between the 20 and $80 \%$ quantiles (shaded areas) and the means (solid lines) of the out-of-sample performance $J \big ( \widehat { x } _ { N } ( \varepsilon ) \big )$ as a function of $\varepsilon$ estimated using 200 independent simulation runs.We observe that the out-of-sample performance improves (decreases) up to a critical Wasserstein radius $\varepsilon _ { \mathrm { c r i t } }$ and then deteriorates (increases). This stylized fact was observed consistently across all of simulations and provides an empirical justification for adopting a distributionally robust approach.

Figure 5 also visualizes the reliability of the performance guarantees offered by our distributionally robust portfolio model. Specifically, the dashed lines represent the empirical probability of the event $J \left( \widehat { x } _ { N } ( \varepsilon ) \right) \dot { \leq } \widehat { J } _ { N } ( \varepsilon )$ with respect to 200 independent training datasets.We find that the reliability is nondecreasing in $\varepsilon$ . This observation has intuitive appeal because ${ \widehat { J } } _ { N } ( \varepsilon ) \geq J ( { \widehat { x } } _ { N } { \widehat { ( } } \varepsilon ) )$ whenever $\mathbb { P } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } )$ , and the latter event becomes increasingly likely as $\varepsilon$ grows. Figure 5 also indicates that the certificate guarantee sharply rises towards 1 near the critical Wasserstein radius $\varepsilon _ { \mathrm { c r i t } }$ . Hence, the out-of-sample performance of the distributionally robust portfolios improves as long as the reliability of the performance guarantee is noticeably smaller than 1 and deteriorates when it saturates at 1.Even though this observation was made consistently across all simulations,we were unable to validate it theoretically.

# 7.2.2 Portfolios driven by out-of-sample performance

$\varepsilon$ ${ \widehat { x } } _ { N } ( \varepsilon )$ Trhat $J ( \widehat { x } _ { N } ( \varepsilon ) )$ $\widehat { \varepsilon } _ { N } ^ { \mathrm { o p t } }$

minimizes $J ( \widehat { x } _ { N } ( \varepsilon ) )$ over all $\varepsilon \ge 0$ note that $\widehat { \varepsilon } _ { N } ^ { \mathrm { o p t } }$ inherits the dependence on the training data from $J ( \widehat { x } _ { N } ( \varepsilon ) )$ . As the true distribution $\mathbb { P }$ is unknown, however, it is impossible to evaluate and minimize $J ( \widehat { x } _ { N } ( \varepsilon ) )$ . In practice,the best we can hope for is to approximate $\widehat { \varepsilon } _ { N } ^ { \mathrm { o p t } }$ usingthertaatiss accomplish this goal:

· Holdout method: Partition $\widehat { \xi } _ { 1 } , \dots , \widehat { \xi } _ { N }$ into a training dataset of size $N _ { T }$ and a validation dataset of size $N _ { V } = N - N _ { T }$ Using only the training dataset, solve (27) for alarge but finite number of candidate radii $\varepsilon$ to obtain $\widehat { x } _ { N _ { T } } ( \varepsilon )$ . Use the validation   
approximation.Set dataset to estimate the out-of-sample performance of $\widehat { \varepsilon } _ { N } ^ { \mathrm { h m } }$ to any $\varepsilon$ thamr $\widehat { x } _ { N _ { T } } ( \varepsilon )$ via the sample average $\widehat { x } _ { N } ^ { \mathrm { h m } } =$   
$\widehat { x } _ { N _ { T } } ( \widehat { \varepsilon } _ { N } ^ { \mathrm { h m } } )$ as hedata-driven solutinand $\widehat { J } _ { N } ^ { \mathrm { h m } } = \widehat { J } _ { N _ { T } } ( \widehat { \varepsilon } _ { N } ^ { \mathrm { h m } } )$ as the corrsponding certificate.   
$k$ -fold cross validation: Partition $\widehat { \xi } _ { 1 } , \dots , \widehat { \xi } _ { N }$ into $k$ subsets,and run the holdout method $k$ times.In each run, use exactly one subset as the validation dataset and merge the remaining $k - 1$ subsets to a training dataset. Set $\widehat { \varepsilon } _ { N } ^ { \mathrm { c v } }$ to the average of $k$ $\varepsilon = \widehat { \varepsilon } _ { N } ^ { \mathrm { c v } }$ $N$ ${ \widehat { x } } _ { N } ^ { \mathrm { c v } } = { \widehat { x } } _ { N } ( { \widehat { \varepsilon } } _ { N } ^ { \mathrm { c v } } )$ $\widehat { J } _ { N } ^ { \mathrm { c v } } = \widehat { J } _ { N } ( \widehat \varepsilon _ { N } ^ { \mathrm { c v } } )$ as the corresponding certificate.

The holdout method is computationally cheaper, but cross validation has superior statistical properties. There are several other methods to estimate the best Wassertein radius $\widehat { \varepsilon } _ { N } ^ { \mathrm { o p t } }$ . By construction, however, no method can provide aradius $\widehat { \varepsilon } _ { N }$ such that $\widehat { x } _ { N } ( \widehat { \varepsilon } _ { N } )$ hasabeterouofsaplepfoe than $\widehat { x } _ { N } ( \widehat { \varepsilon } _ { N } ^ { \mathrm { o p t } } )$

In all experiments we compare the distributionally robust approach based on the Wasserstein ambiguity set with the classical sample average approximation (SAA) and with a state-of-the-art data-driven distributionally robust approach,where the ambiguity set is defined via a linear-convex ordering (LCX)-based goodness-of-fit test [7, Section 3.3.2]. The size of the LCX ambiguity set is determined by a single parameter, which should be tuned to optimize the out-of-sample performance.While the best parameter value is unavailable, it can again be estimated using the holdout method or via cross validation. To our best knowledge, the LCX approach represents the only existing data-driven distributionally robust approach for continuous uncertainty spaces that enjoys strong finite-sample guarantees,asymptotic consistency as well as computational tractability.4

To keep the computational burden manageable, in all experiments we select the Wasserstein radius as well as the LCX size parameter from within the discrete set $\mathcal { E } = \{ \varepsilon = b \cdot 1 0 ^ { c } : b \in \{ 0 , . . . , 9 \}$ ， $c \in \{ - 3 , - 2 , - 1 \} \}$ instead of $\mathbb { R } _ { + }$ . We have verified that refining or extending $\mathcal { E }$ has only a marginal impact on our results, which indicates that $\mathcal { E }$ provides a sufficiently rich approximation of $\mathbb { R } _ { + }$

In Fig. 6a-c the sizes of the (LCX and Wasserstein) ambiguity sets are determined via the holdout method, where $80 \%$ of the data are used for training and $20 \%$ for validation. Figure 6a visualizes the tube between the 2O and $80 \%$ quantiles (shaded areas)as well as the mean value (solid lines) of the out-of-sample performance $J ( \widehat { x } _ { N } )$ as a function of the sample size $N$ and based on 2OO independent simulation runs, where $\widehat { x } _ { N }$ is set to the minimizer of the SAA (blue), LCX (purple) and Wasserstein (green) problems, respectively. The constant dashed line represents the optimal value $J ^ { \star }$ of the original stochastic program (1), which is computed through an SAA problem with $N = 1 0 ^ { 6 }$ samples. We observe that the Wasserstein solutions tend to be superior to the SAA and LCX solutions in terms of out-of-sample performance.

![](images/eb17f9f74ebaab6b1eb8602550fdb86f63af778c275f782d735b05b2a5dce2b6.jpg)  
Fig. 6 Out-of-sample performance $J ( \widehat { x } _ { N } )$ , certificate $\widehat { J } _ { N }$ ,and certificate reliability $\mathbb { P } ^ { N } \big [ J ( \widehat { x } _ { N } ) \leq \widehat { J } _ { N } \big ]$ for the performance-driven SAA,LCX and Wasserstein solutions as a function of $N$ .(a) Holdout method, (b) Holdout method,(c) Holdout method, (d) $k$ -fold cross validation, (e) $k$ -fold cross validation, (f) $k$ -fold cross validation, $\mathbf { \tau } ( \mathbf { g } )$ optimal size, $\mathbf { ( h ) }$ optimal size,(i) optimal size (color figure online)

Figure 6b shows the optimal values $\widehat { J _ { N } }$ of the SAA,LCX and Wasserstein problems, where the sizes of the ambiguity sets are chosen via the holdout method. Unlike Fig. 6a, Fig. 6b thus reports in-sample estimates of the achievable portfolio performance. As expected, the SAA approach is over-optimistic due to the optimizer's curse, while the LCX and Wasserstein approaches err on the side of caution. All three methods are known to enjoy asymptotic consistency, which is in agreement with all in-sample and out-of-sample results.

Figure 6c visualizes the reliability of the different performance certificates, that is, the empirical probability of the event $J ( \widehat { x } _ { N } ) \le \widehat { J } _ { N }$ evaluated over 2OO independent simulation runs. Here, $\widehat { x } _ { N }$ represents either an optimal portfolio of the SAA, LCX or Wasserstein problems，while $\widehat { J _ { N } }$ denotes the corresponding optimal value. The optimal SAA portfolios display a disappointing out-of-sample performance relative to the optimistically biased mimimum of the SAA problem—particularly when the training data is scarce.In contrast, the out-of-sample performance of the optimal LCX and Wasserstein portfolios often undershoots $\widehat { J _ { N } }$

Figure 6d-f show the same graphs as Fig. 6a-c, but now the sizes of the ambiguity sets are determined via $k$ -fold cross validation with $k = 5$ . In this case, the out-ofsample performance of both distributionally robust methods improves slightly, while the corresponding certificates and their reliabilities increase significantly with respect to the naive holdout method. However, these improvements come at the expense of a $k$ -fold increase in the computational cost.

One could think of numerous other statistical methods to select the size of the Wasserstein ambiguity set. As discussed above, however, if the ultimate goal is to minimize the out-of-sample performance of ${ \widehat { x } } _ { N } ( \varepsilon )$ , then the best possible choice is $\varepsilon = \widehat { \varepsilon } _ { N } ^ { \mathrm { { o p t } } }$ . Similarly,onecan construct a size parameter fortheLCX ambiguity set that leads to the best possible out-of-sample performance of any LCX solution. We emphasize that these optimal Wasserstein radii and LCX size parameters are not available in practice because computing $J ( \widehat { x } _ { N } ( \varepsilon ) )$ requires knowledge of the data-generating distribution. In our experiments we evaluate $J ( \widehat { x } _ { N } ( \varepsilon ) )$ to high accuracy for every fixed $\varepsilon \in { \mathcal { E } }$ using $2 \cdot 1 0 ^ { 5 }$ validation samples,which are independent from the (much fewer) training samples used to compute ${ \widehat { x } } _ { N } ( \varepsilon )$ . Figure $6 \mathrm { g - i }$ show the same graphs as Fig. 6a-c for optimally sized ambiguity sets.By construction, no method for sizing the Wasserstein or LCX ambiguity sets can result in a better out-of-sample performance, respectively. In this sense, the graphs in Fig. 6g capture the fundamental limitations of the different distributionally robust schemes.

# 7.2.3 Portfolios driven by reliability

In Sect. 7.2.2 the Wasserstein radii and LCX size parameters were calibrated with the goal to achieve the best out-of-sample performance. Figure 6c, f, i reveal, however, that by optimizing the out-of-sample performance one may sacrifice reliability. An alternative objective more in line with the general philosophy of Sect. 2 would be to choose Wasserstein radii that guarantee a prescribed reliability level. Thus, for a given $\beta \in [ 0 , 1 ]$ we should find the smallest Wasserstein radius $\varepsilon \ge 0$ for which the optimal value ${ \widehat { J } } _ { N } ( \varepsilon )$ of（27） provides an upper $1 - \beta$ confidence bound on the out-of-sample performance $J ( \widehat { x } _ { N } ( \varepsilon ) )$ of its optimal solution. As the true distribution $\mathbb { P }$ is unknown, however, the optimal Wasserstein radius corresponding to a given $\beta$ cannot be computed exactly. Instead, we must derive an estimator ${ \widehat { \varepsilon } } _ { N } ^ { \beta }$ that depends on ${ \widehat { \varepsilon } } _ { N } ^ { \beta }$ and the corresponding reliability-driven portfolio via bootstrapping as follows:

(1） Construct $k$ resamples of size $N$ (with replacement) from the original training dataset. It is well known that, as $N$ grows, the probability that any fixed training data point appears in a particular resample converges to $\textstyle { \frac { e - 1 } { e } } \approx { \frac { 2 } { 3 } }$ . Thus, about $\frac { N } { 3 }$ training samples are absent from any resample. We collect all unused samples in a validation dataset.

(2） For each resample $\kappa = 1 , \ldots , k$ and $\varepsilon \ge 0$ , solve problem (27) using the Wasserstein ball of radius $\varepsilon$ around theempirical distribution $\widehat { \mathbb { P } } _ { N } ^ { \kappa }$ on the $\kappa$ -th resample. The resulting optimal decision and optimal value are denoted as ${ \widehat { x } } _ { N } ^ { \kappa } ( \varepsilon )$ and $\widehat { J _ { N } ^ { \kappa } } ( \varepsilon )$ ， respectively. Next, estimate the out-of-sample performance $J ( \widehat { x } _ { N } ^ { \kappa } ( \varepsilon ) )$ of ${ \widehat { x } } _ { N } ^ { \kappa } ( \varepsilon )$ using the sample average over the $\kappa$ -th validation dataset.

(3) Set $\widehat { \varepsilon } _ { N } ^ { \beta }$ to the smallest $\varepsilon \geq 0$ so that the certificate $\widehat { J _ { N } ^ { k } } ( \varepsilon )$ exceeds the estimate of $J ( \widehat { x } _ { N } ^ { \kappa } ( \varepsilon ) )$ in at least $( 1 - \beta ) \times k$ different resamples.

(4） Compute the data-driven portfolio ${ \widehat { x } } _ { N } = { \widehat { x } } _ { N } ( { \widehat { \varepsilon } } _ { N } ^ { \beta } )$ and the corresponding certificate $\widehat { J _ { N } } = \widehat { J _ { N } } ( \widehat { \varepsilon } _ { N } ^ { \beta } )$ using the original training dataset.

As in Sect. 7.2.2, we compare the Wasserstein approach with the LCX and SAA approaches. Specifically, by using bootstrapping,we calibrate the size of the LCX ambiguity set so as to guarantee a desired reliability level $1 - \beta$ . The SAA problem, on the other hand, has no free parameter that can be tuned to meet a prescribed reliability target. Nevertheless, we can construct a meaningful certificate of the form $\begin{array} { r } { \widehat { J } _ { N } ( \Delta ) \mathrel { \mathop : } = \widehat { J } _ { \mathrm { S A A } } ^ { \sim } + \Delta } \end{array}$ for the SAA portfolio by adding a non-negative constant to the optimal value of the SAA problem. Our aim is to find the smallest offset $\Delta \geq 0$ with the property that $\widehat { J } _ { N } ( \Delta )$ provides an upper $1 - \beta$ confidence bound on the out-ofsample performance $J ( \widehat { x } _ { \mathrm { S A A } } )$ of the optimal SAA portfolio $\widehat { x } _ { \mathrm { S A A } }$ . The optimal offset corresponding to a given $\beta$ cannot be computed exactly. Instead, we must derive an estimator $\widehat { \Delta } _ { N } ^ { \beta }$ that depends on the training data. Such an estimator can be found through a simple variant of the above bootstrapping procedure.

In all experiments we set the number of resamples to $k = 5 0$ .Figure $\operatorname { 7 a - c }$ visualize the out-of-sample performance,the certificate and the empirical reliability of the reliability-driven portfolios obtained with the SAA,LCX and Wasserstein approaches, respectively, for the reliability target $1 - \beta = 9 0 \%$ and based on 2OO independent simulation runs. Figure 7d-f show the same graphs as Fig. 7a-c but for the reliability target $1 - \beta = 7 5 \%$ . We observe that the new SAA certificate now overestimates the true optimal value of the portfolio problem. Moreover, while the empirical reliability of the SAA solution now closely matches the desired reliability target, the empirical reliabilities of the LCX and Wasserstein solutions are similar but noticeably exceed the prescribed reliability threshold.A possible explanation for this phenomenon is that the $k$ resamples generated by the bootstrapping algorithm are not independent, which may give rise to a systematic bias in estimating the Wasserstein radii required for the desired reliability levels.

# 7.2.4 Impact of the sample size on the Wasserstein radius

It is instructive to analyze the dependence of the Wasserstein radii on the sample size $N$ for different data-driven schemes. As for the performance-driven portfolios from Sect. 7.2.2, Fig. 8 depicts the best possible Wasserstein radius $\widehat { \varepsilon } _ { N } ^ { \mathrm { o p t } }$ as well as the Wasserstein radii $\widehat { \varepsilon } _ { N } ^ { \mathrm { h m } }$ and $\widehat { \varepsilon } _ { N } ^ { \mathrm { c v } }$ obtained bythe holdout method and via $k$ -fold cross validation, respectively. As for the reliability-driven portfolios from Sect. 7.2.3,Fig. 8 furtherdepictsthe Wassersteinradi $\widehat { \varepsilon } _ { N } ^ { \beta }$ for $\beta \in \{ 1 0 \% , 2 5 \% \}$ , obtained by boostrapping. All results are averaged across 2OO independent simulation runs. As expected from Theorem 3.6, all Wasserstein radii tend to zero as $N$ increases. Moreover, the convergence rate is approximately equal to $N ^ { - { \frac { 1 } { 2 } } }$ . This rate is likely to be optimal.

![](images/5ea38fdb19a9b6886d3424b4877abf7399e17cbbe215a9fff963d07fa9585f00.jpg)  
Fig.7 Out-of-sample performance $J ( \widehat { x } _ { N } )$ ,certificate $\widehat { J } _ { N }$ ,and certificate reliability $\mathbb { P } ^ { N } \big [ J ( \widehat { x } _ { N } ) \leq \widehat { J } _ { N } \big ]$ for the reliability-driven SAA,LCX and Wasserstein portfolios as a function of $N$ .(a) $\beta = 1 0 \%$ ， $( { \bf b } ) \beta = 1 0 \%$ ， (c) $\beta = 1 0 \%$ ，(d) $\beta = 2 5 \%$ (e) $\beta = 2 5 \%$ (f) $\beta = 2 5 \%$

![](images/98bcaaa21ceba1cd98f5f2421a6491bb2be2fae758628cd7ba3798fe3ad1f83c.jpg)  
Fig 8Optimalperforn nce-diven assesi ass ${ \widehat { \varepsilon } } _ { N } ^ { \mathrm { o p t } }$ and its esimates $\widehat { \varepsilon } _ { N } ^ { \mathrm { h m } }$ and $\widehat { \varepsilon } _ { N } ^ { \mathrm { c v } }$ obtained yi $k$ -fold cross validation, respectively,as well as the reliability-driven Wasserstein radius $\widehat { \varepsilon } _ { N } ^ { \beta }$ for $\beta \in \{ 1 0 \% , 2 5 \% \}$ obtained via bootstrapping

Indeed, if $\mathbb { X }$ is a singleton, then every quantile of the sample average estimator $\widehat { J } _ { \mathrm { S A A } }$ converges to $J ^ { \star }$ at rate $N ^ { - \frac { 1 } { 2 } }$ due to the central limit theorem. Thus, if $\widehat { \varepsilon } _ { N } = o ( N ^ { - \frac { 1 } { 2 } } )$ ， then ${ \widehat { J } } _ { N }$ also converges to $J ^ { \star }$ at leading order $N ^ { - \frac { 1 } { 2 } }$ by Theorem 6.3, which applies as the loss function is convex.This indicates that the a priori rate $N ^ { - { \frac { 1 } { m } } }$ suggested by Theorem 3.4 is too pessimistic in practice.

# 7.3 Simulation results: uncertainty quantification

Investors often wish to determine the probability that a given portfolio will outperform various benchmark indices or assets. Our results on uncertainty quantification developed in Sect. 5.2 enable us to compute this probability in a meaningful way—solely on the basis of the training dataset.

Assume for example that we wish to quantify the probability that any data-driven portfolio $\widehat { x } _ { N }$ outperforms the three most risky assets in the market jointly. Thus, we should compute the probability of the closed polytope

$$
\widehat { \mathbb { A } } = \left\{ \xi \in \mathbb { R } ^ { m } : \left. \widehat { x } _ { N } , \xi \right. \geq \xi _ { i } \forall i = 8 , 9 , 1 0 \right\} .
$$

As the true distribution $\mathbb { P }$ is unknown, the probability $\mathbb { P } [ \xi \in \widehat { \mathbb { A } } ]$ cannot be evaluated exactly. Note that $\widehat { \mathbb { A } }$ as well as $\mathbb { P } [ \xi \in \widehat { \mathbb { A } } ]$ constitute random objects that depend on $\widehat { x } _ { N }$ and thus on the training data. Using the same training dataset that was used to compute $\widehat { x } _ { N }$ , however, we may estimate $\mathbb { P } [ \boldsymbol { \xi } \in \widehat { \mathbb { A } } ]$ from above and below by

$$
\operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \big [ \xi \in \widehat { \mathbb { A } } \big ] \qquad \mathrm { a n d } \qquad \operatorname* { i n f } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \big [ \xi \in \widehat { \mathbb { A } } \big ] = 1 - \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \big [ \xi \not \in \widehat { \mathbb { A } } \big ] ,
$$

respectively. Indeed,recallthat the true data-generating probability distribution resides in the Wasserstein ball of radius $\varepsilon _ { N } ( \beta )$ defined in (8) with probability $1 - \beta$ . Therefore, we have

$$
\begin{array} { r l } & { 1 - \beta \leq \mathbb { P } ^ { N } \bigg [ \widehat { \Xi } _ { N } : \mathbb { P } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } ) \bigg ] } \\ & { \quad \leq \mathbb { P } ^ { N } \bigg [ \widehat { \Xi } _ { N } : \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \big [ \mathbb { A } \big ] \geq \mathbb { P } \big [ \mathbb { A } \big ] \forall \mathbb { A } \in \mathfrak { B } ( \Xi ) \bigg ] } \\ & { \quad = \mathbb { P } ^ { N } \bigg [ \widehat { \Xi } _ { N } : \operatorname* { i n f } _ { \mathbb { A } \in \mathfrak { B } ( \Xi ) } \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \big [ \mathbb { A } \big ] - \mathbb { P } \big [ \mathbb { A } \big ] \geq 0 \bigg ] , } \end{array}
$$

where $\mathfrak { B } ( \Xi )$ denotes the set of all Borel subsets of $\Xi$ . The data-dependent set $\widehat { \mathbb { A } } _ { N }$ can now be viewed as a (measurable) mapping from $\widehat { \Xi } _ { N }$ to the subsets in $\mathfrak { B } ( \Xi )$ . The above inequality then implies

$$
\mathbb { P } ^ { N } \bigg [ \widehat { \Xi } _ { N } : \operatorname* { s u p } _ { \mathbb { Q } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } ) } \mathbb { Q } \big [ \widehat { \mathbb { A } } _ { N } \big ] - \mathbb { P } \big [ \widehat { \mathbb { A } } _ { N } \big ] \geq 0 \bigg ] \geq 1 - \beta .
$$

![](images/60b60f26b0b7ab87eff89ee5cc417203ce70002761c9ee7af8b2e75f60b17b75.jpg)  
Fig. 9 Excess $\widehat { J } _ { N } ^ { + } ( \varepsilon ) - \mathbb { P } \widehat { \mathbb { A } } ]$ and shortfall $\widehat { J } _ { N } ^ { - } ( \varepsilon ) - \mathbb { P } [ \widehat { \mathbb { A } } ]$ (solid lines,left axis）as wellas reliability $\mathbb { P } ^ { N } [ \widehat { J } _ { N } ^ { - } ( \varepsilon ) \leq \mathbb { P } [ \widehat { \mathbb { A } } ] \leq \widehat { J } _ { N } ^ { + } ( \varepsilon ) ]$ (dashed i $\varepsilon$ .(a) $N = 3 0$ (b) $N = 3 0 0$

Thus, $\operatorname* { s u p } \{ \mathbb { Q } [ { \widehat { \mathbb { A } } } _ { N } ] : \mathbb { Q } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( { \widehat { \mathbb { P } } } _ { N } ) \}$ provides indeed an upper bound on $\mathbb { P } [ \widehat { \mathbb { A } } _ { N } ]$ with confidence $1 - \beta$ . Similarly, one can show that inf $\{ \mathbb { Q } [ \widehat { \mathbb { A } } _ { N } ] : \mathbb { Q } \in \mathbb { B } _ { \varepsilon _ { N } ( \beta ) } ( \widehat { \mathbb { P } } _ { N } ) \}$ provides a lower confidence bound on $\mathbb { P } [ \widehat { \mathbb { A } } _ { N } ]$

The upper confidence bound can be computed by solving the linear program (17a). Replacing $\widehat { \mathbb { A } }$ with its interior in the lower confidence bound leads to another(potentially weaker) lower bound that can be computed by solving the linear program (17b). We denote these computable bounds by $\hat { J } _ { N } ^ { + } ( \varepsilon )$ and $\widehat { J } _ { N } ^ { - } ( \varepsilon )$ , respectively. In all subsequent experiments $\widehat { x } _ { N }$ is set to a solution of the distributionally robust program (27) calibrated via $k$ -fold cross validation as described in Sect. 7.2.2.

# 7.3.1 Impact of the Wasserstein radius

As $\widehat { J } _ { N } ^ { + } ( \varepsilon )$ and $\widehat { J } _ { N } ^ { - } ( \varepsilon )$ estimate a random target $\mathbb { P } [ \widehat { \mathbb { A } } ]$ , it makes sense to filter out the randomness of the target and to study only the differences $\widehat { J } _ { N } ^ { + } ( \varepsilon ) - \mathbb { P } [ \widehat { \mathbb { A } } ]$ and $\widehat { J _ { N } ^ { - } } ( \varepsilon ) - \mathbb { P } [ \widehat { \mathbb { A } } ]$ . Figure 9a,b visualize the empirical mean (solid lines)as well as the tube between the empirical 20 and $80 \%$ quantiles (shaded areas) of these differences as a function of the Wasserstein radius $\varepsilon$ ,based on 2OO training datasets of cardinality $N = 3 0$ and $N = 3 0 0$ , respectively. Figure 9 also shows the empirical reliability of the bounds (dashed lines), that is, the empirical probability of the event $\begin{array} { r } { \widehat { J } _ { N } ^ { - } ( \varepsilon ) \leq \dot { \mathbb { P } } [ \widehat { \mathbb { A } } ] \leq } \end{array}$ $\widehat { J } _ { N } ^ { + } ( \varepsilon )$ . Note that the reliability drops to O for $\varepsilon = 0$ , in which case both $\widehat { J } _ { N } ^ { + } ( 0 )$ and $\widehat { J } _ { N } ^ { - } ( 0 )$ coincide with the SAA estimator for $\mathbb { P } [ \widehat { \mathbb { A } } ]$ .Moreover, at $\varepsilon = 0$ the set $\widehat { \mathbb { A } }$ is constructed from the SAA portfolio $\widehat { x } _ { N }$ , whose performance is overestimated on the training dataset. Thus, the SAA estimator for $\mathbb { P } [ \widehat { \mathbb { A } } ]$ , which is evaluated using the same training dataset, is positively biased. For $\varepsilon > 0$ , finally, the reliability increases as the shaded confidence intervals move away from 0.

# 7.3.2 Impact of the sample size

We propose a variant of the $k$ -fold cross validation procedure for selecting $\varepsilon$ in uncertainty quantification. Partition $\widehat { \xi } _ { 1 } , \dots , \widehat { \xi } _ { N }$ into $k$ subsets and repeat the following holdout method $k$ times. Select one of the subsets as the validation set of size $N _ { V }$ and merge the remaining $k - 1$ subsets to a training dataset of size $N _ { T } = N - N _ { V }$ Use the validation set to compute the SAA estimator of $\mathbb { P } [ \widehat { \mathbb { A } } ]$ , and use the training dataset to compute $\widehat { J } _ { N _ { T } } ^ { + } \left( \varepsilon \right)$ frgebe $\varepsilon$ Set $\widehat { \varepsilon } _ { N } ^ { \mathrm { h m } }$ to the smallest candidate radius for which the SAA estimator of $\mathbb { P } [ \widehat { \mathbb { A } } ]$ is not larger than $\widehat { J } _ { N _ { T } } ^ { + } ( \varepsilon )$ . Next, set $\widehat { \varepsilon } _ { N } ^ { \mathrm { c v } }$ to the average of the Wasserstein radii obtained from the $k$ holdout runs, and report $\widehat { J } _ { N } ^ { + } = \widehat { J } _ { N } ^ { + } ( \widehat { \varepsilon } _ { N } ^ { \mathrm { c v } } )$ as the data-driven upper bound on $\mathbb { P } [ \widehat { \mathbb { A } } ]$ The data-driven lower bound $\widehat { J } _ { N } ^ { - }$ is constructed analogously in the obvious way.

![](images/0a4cad4541cd6fdb838a4edf6b7cb340ea0f7ea8c60e48a20a4d751d30dba55d.jpg)  
Fig.10 Dependence of the confidence bounds and the Wasserstein radius on $N$ . (a) Excess $\widehat { J } _ { N } ^ { + } - \mathbb { P } [ \widehat { \mathbb { A } } ]$ and shortfall $\widehat { J } _ { N } ^ { - } - \mathbb { P } [ \widehat { \mathbb { A } } ]$ of the data-driven confidence bounds for $\mathbb { P } [ \widehat { \mathbb { A } } ]$ . (b) Data-driven Wasserstein radius $\widehat { \varepsilon } _ { N } ^ { \mathrm { c v } }$ obtained via $k$ fold crossvalidation

Figure 1Oa visualizes the empirical means (solid lines) as well as the tubes between the empirical 20 and $80 \%$ quantiles (shaded areas) of $\widehat { J _ { N } ^ { + } } - \mathbb { P } [ \widehat { \mathbb { A } } ]$ and $\widehat { J } _ { N } ^ { - } - \mathbb { P } [ \widehat { \mathbb { A } } ]$ as a function of the sample size $N$ , based on 300 independent training datasets. As expected, the confidence intervals shrink and converge to O as $N$ increases. We emphasize that $\widehat { J } _ { N } ^ { + }$ and $\widehat { J } _ { N } ^ { - }$ are computed solely on the basis of $N$ training samples, whereas the computation of $\mathbb { P } [ \widehat { \mathbb { A } } ]$ necessitates a much larger dataset, particularly if $\widehat { \mathbb { A } }$ constitutes a rare event.

Figure 10b shows the Wasserstein radius $\widehat { \varepsilon } _ { N } ^ { \mathrm { c v } }$ obtained via $k$ -fold cross validation (both for $\widehat { J _ { N } ^ { + } }$ and $\widehat { J } _ { N } ^ { - } .$ ). As usual, all results are averaged across 30O independent simulation runs. A comparison with Fig. 8 reveals that the data-driven Wasserstein radii in uncertainty quantification display a similar but faster polynomial decay than in portfolio optimization.We conjecture that this is due to the absence of decisions, which implies that uncertainty quantification is less susceptible to the optimizer's curse.Thus, nature (i.e.,the fictitious adversary choosing the distribution in the ambiguity set) only has to compensate for noise but not for bias. A smaller Wasserstein radius seems to be sufficient for this purpose.

Acknowledgements We thank Soroosh Shafieezadeh Abadeh for helping us with the numerical experiments.The authors are grateful to Vishal Gupta, Ruiwei Jiang and Nathan Kallus for their valuable comments. This research was supported by the Swiss National Science Foundation under Grant BSCGI0_157733.

Open Access This article is distributed under the terms of the Creative Commons Atribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/), which permits unrestricted use, distribution, and reproduction in any medium, provided you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license,and indicate if changes were made.

# Appendix A

The following technical lemma on the pointwise approximation of an upper semicontinuous function by a non-increasing sequence of Lipschitz continuous majorants strengthens [31, Theorem 4.2], which focuses on bounded domains and continuous (but not necessarily Lipschitz continuous) majorants.

Lemma A.1 If $h : \Xi \to \mathbb { R }$ is upper semicontinuous and satisfies $h ( \xi ) \leq L ( 1 + \| \xi \| )$ for some $L \geq 0$ ， then there exists a non-increasing sequence of Lipschitz continuous functions that converge pointwise to $h$ on E.

Proof The proof is constructive. Define the functions

$$
h _ { k } ( \xi ) = \operatorname* { s u p } _ { \xi ^ { \prime } \in \Xi } h ( \xi ^ { \prime } ) - k L \| \xi - \xi ^ { \prime } \| , \quad k \in \mathbb { N } ,
$$

where $L$ is the linear growth rate of $h$ .Note that by construction $h _ { k } ( \xi ) \leq L ( 1 { + } \| \xi \| )$ .As $\xi ^ { \prime } = \xi$ is feasible in the maximization problem defining $h _ { k } ( \xi )$ , we have $h _ { k } ( \xi ) \geq h ( \xi )$ for all $\xi \in \Xi$ and $k \in \mathbb N$ Moreover, $h _ { k } ( \xi )$ is Lipschitz continuous with Lipschitz constant $k L$ (as $h _ { k } ( \xi )$ constitutes a supremum of norm functions with this property). Given any $\xi \in \Xi$ , it remains to be shown that $\operatorname* { l i m } _ { k \to \infty } h _ { k } ( \xi ) = h ( \xi )$ . Thus, choose $\xi _ { k } ^ { \prime } \in \Xi$ with

$$
h _ { k } ( \xi ) = \operatorname* { s u p } _ { \xi ^ { \prime } \in \Xi } h ( \xi ^ { \prime } ) - k L \| \xi - \xi ^ { \prime } \| \leq h ( \xi _ { k } ^ { \prime } ) - k L \| \xi - \xi _ { k } ^ { \prime } \| + \frac { 1 } { k } .
$$

We first show that $\xi _ { k }$ converges to $\xi$ as $k$ tends to $\infty$ . Indeed, we have

$$
\begin{array} { l } { { h ( \xi ) \leq h _ { k } ( \xi ) \leq h ( \xi _ { k } ^ { \prime } ) - k L \| \xi - \xi _ { k } ^ { \prime } \| + \displaystyle \frac { 1 } { k } \leq L ( 1 + \| \xi _ { k } ^ { \prime } \| ) - k L \| \xi - \xi _ { k } ^ { \prime } \| + \displaystyle \frac { 1 } { k } } } \\ { { \mathrm { } \leq L ( 1 + \| \xi - \xi _ { k } ^ { \prime } \| + \| \xi \| ) - k L \| \xi - \xi _ { k } ^ { \prime } \| + \displaystyle \frac { 1 } { k } = L ( 1 + \| \xi \| ) + \displaystyle \frac { 1 } { k } } } \\ { { \mathrm { } - ( k - 1 ) L \| \xi - \xi _ { k } ^ { \prime } \| , } } \end{array}
$$

which implies

$$
\| \xi - \xi _ { k } ^ { \prime } \| \leq \frac { 1 } { L ( k - 1 ) } \left( h ( \xi ) - L ( 1 + \| \xi \| ) - \frac { 1 } { k } \right) ,
$$

that is, $\| \xi - \xi _ { k } ^ { \prime } \| \to 0$ as $k \to \infty$ . Therefore, we find

$$
h ( \xi ) \leq \operatorname* { l i m } _ { k \to \infty } h _ { k } ( \xi ) \leq \operatorname* { l i m } _ { k \to \infty } h ( \xi _ { k } ^ { \prime } ) - k L \| \xi - \xi _ { k } ^ { \prime } \| + \frac { 1 } { k } \leq \operatorname* { l i m } _ { k \to \infty } h ( \xi _ { k } ^ { \prime } ) \leq h ( \xi ) ,
$$

where the last inequality is due to the upper semicontinuity of $h$ . This concludes the proof. □

# References

1.Ben-Tal,A,den Hertog,D., Vial,J.-P.: Deriving robustcounterparts of nonlinearuncertain inequalities. Math. Program.149,265-299 (2015)   
2.Ben-Tal, A. den Hertog, D.，Waegenaere, A.D., Melenberg, B.，Rennen, G.: Robust solutions of optimization problems affected by uncertain probabilities. Manag. Sci. 59,341-357 (2013)   
3.Ben-Tal, A.,ElGhaoui,L.,Nemirovski,A.: Robust Optimization.Princeton University Press,Princeton (2009)   
4.Bertsekas, D.P.: Convex Optimization Theory. Athena Scientific, Belmont (2009)   
5.Bertsekas, D.P.: Convex Optimization Algorithms. Athena Scientific, Belmont (2015)   
6.Bertsimas,D.,Doan, X.V., Natarajan,K.,Teo, C.-P.: Models for minimax stochastic linearoptimization problems with risk aversion. Math. Oper. Res.35,580-602 (2010)   
7.Bertsimas,D., Gupta, V., Kallus, N.: Robust SAA. Available at arXiv:1408.4445 (2014)   
8. Bertsimas,D., Popescu,L: On the relation between option and stock prices: aconvex optimization approach. Oper. Res.50,358-374 (2002) 9. Bertsimas, D., Sim, M.: The price of robustness. Oper. Res. 52,35-53 (2004)   
10.Boissard, E.: Simple bounds for convergence of empirical and occupation measures in 1-Wasserstein distance. Electron. J. Probab.16,2296-2333(2011)   
11.Bolly,F.,Guilin,A., Villani, C.: Quantitative concentration inequalities for empirical measurson non-compact spaces. Probab. Theory Relat. Fields 137,541-593 (2007)   
12. Boyd, S., Vandenberghe,L.: Convex Optimization. Cambridge University Press, Cambridge(2009)   
13.Brownlees,C., Joly,E.,Lugosi, G.: Empirical risk minimization for heavy-tailed losses.Ann.Stat. 43, 2507-2536 (2015)   
14. Calafiore, G.C.: Ambiguous risk measures and optimal robust portfolios. SIAM J. Optim.18, 853-877 (2007)   
15. Catoni, O.: Challenging the empirical mean and empirical variance: a deviation study. Annales de PInstitut Henri Poincaré,Probabilites et Statistiques 48,1148-1185 (2012)   
16.Chehrazi,N., Weber,T.A.: Monotone approximation of decision problems. Oper. Res.58,1158-1177 (2010)   
17.del Bario,E., Cuesta-Albertos,JA.,Matran,C.,etal.: Tests of goodness offit basedon the $l _ { 2 }$ Wasserstein distance. Ann. Stat. 27,1230-1239 (1999)   
18.Delage,E., Ye, Y.: Distributionally robust optimization under moment uncertainty with application to data-driven problems. Oper. Res. 58, 595-612 (2010)   
19.El Ghaoui,L., Oks, M., Oustry,F.: Worst-case value-at-risk and robust portfolio optimization: a conic programming approach. Oper. Res. 51, 543-556 (2003)   
20. Erdogan, E., Iyengar, G.: Ambiguous chance constrained problems and robust optimization. Math. Program.107,37-61 (2006)   
21.Fournier,N., Guillin, A.: On the rate of convergence in Wasserstein distance of the empirical measure. Probab. Theory Relat.Fields 162,1-32 (2014)   
22. Goh, J., Sim, M.: Distributionally robust optimization and its tractable approximations. Oper. Res. 58, 902-917 (2010)   
23.Hanasusanto, G.A., Kuhn, D.: Robust data-driven dynamic programming. In: Burges, C.J.C., Bottou, L., Welling M., Ghahramani, Z., Weinberger, K.Q. (eds.） Advances in Neural Information Processng Systems,26: 27th Annual Conference on Neural Information Processing Systems 2013, pp.827-835. Curran Associates, Inc., (2013)   
24.Hanasusanto, G.A., Kuhn,D., Wiesemann, W.: A comment on computational complexity of stochastic programming problems. Math. Program. 159, 557-569 (2016) Available at Optimization Online (2013)   
26.Hu,Z.,Hong,L.J.,So,A.M.-C.: Ambiguous probabilistic programs.Available at Optimization Online (2013)   
27． Jiang,R., Guan, Y.: Data-driven chance constrained stochastic program. Math. Program.158,291-327 (2016)   
28.Kallenberg,O.: Foundations of Modern Probability, Probability and its Applications.Springer, New York (1997)   
29.Kantorovich,L.V.,Rubinshtein,G.S.: Ona spaceof totall aditive functions. Vestn.Leningr. Univ. 13, 52-59 (1958)   
30.Lang, S.: Real and Functional Analysis, 3rd edn. Springer, Berlin (1993)   
31. Mashreghi, J.: Representation Theorems in Hardy Spaces. Cambridge University Press, Cambridge (2009)   
32.Mehrotra, S. Zhang, H.: Models and algorithms for distributionally robust least squares problems. Math. Program.146,123-141 (2014)   
33.Muller, A.: Integral probability metrics and their generating classes of functions. Adv. Appl. Probab. 29,429-443 (1997)   
34.Natarajan, K., Sim, M., Uichanco,J.: Tractable robust expected utility and risk models for portfolio optimization. Math.Financ.20,695-731 (010)   
35.Parikh,N.,Boyd,S.: Block splitting for distributed optimization. Math. Program. Comput. 6,77-102 (2014)   
36.Pflug,G.C., Pichler,A.: Multistage Sochastic Optimization.Springer, Berlin (2014)   
37.Pflug, G.C.,Pichler,A.,Wozabal,D.: The1/N investment strategy is optimalunder high modelambiguity. J. Bank. Financ.36,410-417 (2012)   
38.Pflug, G.C., Wozabal, D.: Ambiguity in portfolio selection. Quant. Financ.7, 435-442 (2007)   
39.Postek, K., den Hertog,D., Melenberg,B.: Computationally tractable counterparts of distributionally robust constraints on risk measures. SIAM 58(4), 603-650 (2016). doi:10.1137/151005221   
40. Ramdas, A., Garcia, N., Cuturi, M.: On Wasserstein two sample testing and related families of nonparametric tests. Available at arXiv:1509.02237 (2015)   
41.Rockafelar,R.T., Uryasev, S.: Optimization of conditional value-at-risk. J.Risk 2,21-42 (2000)   
42.Rockafellar,R.T.,Wets,R.J.-B.: Variational Analysis. Springer, Berlin (2010)   
43. Scarf, H.E.: Studies in the mathematical theory of inventory and production. In: Arrow, K.J., Karlin, S., Scarf, H.E.(eds.)AMin-Max Solution of an Inventory Problem,pp.201-209.Stanford University Press, Stanford (1958)   
44.Shapiro,A.: On duality theory of conic linear problems. In: Goberna, M.A.,López,M.A. (eds.)SemiInfinite Programming, pp. 135-165. Kluwer, Boston (2001)   
45.Shapiro,A.: Distributionally robust stochastic programming. Available at Optimization Online (2015)   
46.Shapiro, A., Dentcheva,D., Ruszczyhski, A.: Lectures on Stochastic Programming, 2nd edn,AM (2014)   
47. Shapiro, A., Nemirovski, A.: On complexity of stochastic programming problems. In: Jeyakumar, V., Rubinov, A. (eds.) Continuous Optimization, pp.111-146. Springer, New York (2005)   
48.Smith, J.E., Winkler,R.L.: Theoptimizers curse:Skepticism and postdecision surprise in decision analysis. Manag. Sci. 52,311-322 (2006)   
49.Vapnik, V.N.: Statistical Learning Theory. Wiley, New York (1998)   
50.Villani,C.: Topics inOptimal Transportation. American Mathematical Society, Providence (2003)   
51. Wiesemann, W., Kuhn,D., Sim,M.: Distributionall robust convex optimization. Oper. Res. 62,1358- 1376 (2014)   
52.Wozabal, D.: A framework for optimization under ambiguity. Ann. Oper. Res. 193,21-47 (012)   
53.Wozabal, D.: Robustifying convex risk measures for linear portfolios: a nonparametric approach. Oper. Res. 62,1302-1315 (2014)   
54.Zhao,C.: Data-Driven Risk-Averse Stochastic Program and Renewable Energy Integration. PhD thesis, University of Florida (2014)