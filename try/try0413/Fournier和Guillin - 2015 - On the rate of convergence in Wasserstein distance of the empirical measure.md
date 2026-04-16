![](_page_0_Picture_1.jpeg)

# **On the rate of convergence in Wasserstein distance of the empirical measure**

**Nicolas Fournier · Arnaud Guillin**

Received: 8 December 2013 / Revised: 26 September 2014 / Published online: 18 October 2014 © Springer-Verlag Berlin Heidelberg 2014

**Abstract** Let μ*<sup>N</sup>* be the empirical measure associated to a *N*-sample of a given probability distribution μ on R*<sup>d</sup>* . We are interested in the rate of convergence of μ*<sup>N</sup>* to μ, when measured in the Wasserstein distance of order *p* > 0. We provide some satisfying non-asymptotic *L <sup>p</sup>*-bounds and concentration inequalities, for any values of *<sup>p</sup>* <sup>&</sup>gt; 0 and *<sup>d</sup>* <sup>≥</sup> 1. We extend also the non asymptotic *<sup>L</sup> <sup>p</sup>*-bounds to stationary ρ-mixing sequences, Markov chains, and to some interacting particle systems.

**Keywords** Empirical measure · Sequence of i.i.d. random variables · Wasserstein distance · Concentration inequalities · Quantization · Markov chains · ρ-mixing sequences · Mc Kean-Vlasov particles system

**Mathematics Subject Classification** 60F25 · 60F10 · 65C05 · 60E15 · 65D32

# **1 Introduction and results**

# 1.1 Notation

Let *<sup>d</sup>* <sup>≥</sup> 1 and *<sup>P</sup>*(R*<sup>d</sup>* ) stand for the set of all probability measures on <sup>R</sup>*<sup>d</sup>* . For <sup>μ</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ), we consider an i.i.d. sequence (*Xk* )*k*≥<sup>1</sup> of <sup>μ</sup>-distributed random variables and,

N. Fournier

A. Guillin (B)

Laboratoire de Probabilités et Modèles aléatoires, UMR 7599, UPMC, Case 188, 4 pl. Jussieu, 75252 Paris Cedex 5, France e-mail: nicolas.fournier@upmc.fr

Laboratoire de Mathématiques, UMR 6620, Université Blaise Pascal, Av. des landais, 63177 Aubiere cedex, France e-mail: guillin@math.univ-bpclermont.fr

for *N* ≥ 1, the empirical measure

$$\mu\_N := \frac{1}{N} \sum\_{k=1}^N \delta\_{X\_k} \dots$$

As is well-known, by Glivenko-Cantelli's theorem, μ*<sup>N</sup>* tends weakly to μ as *N* → ∞ (for example in probability, see Van der Vaart-Wellner [\[40\]](#page-31-0) for details and various modes of convergence). The aim of the paper is to quantify this convergence, when the error is measured in some Wasserstein distance. Let us set, for *p* ≥ 1 and μ, ν in *<sup>P</sup>*(R*<sup>d</sup>* ),

$$\mathcal{T}\_p(\mu, \upsilon) = \inf \left\{ \left( \int\_{\mathbb{R}^d \times \mathbb{R}^d} |\mathbf{x} - \mathbf{y}|^p \xi(d\mathbf{x}, d\mathbf{y}) \right) \, : \, \xi \in \mathcal{H}(\mu, \upsilon) \right\},$$

where *<sup>H</sup>*(μ, ν) is the set of all probability measures on <sup>R</sup>*<sup>d</sup>* <sup>×</sup>R*<sup>d</sup>* with marginals <sup>μ</sup> and <sup>ν</sup>. See Villani [\[41\]](#page-31-1) for a detailed study of *<sup>T</sup>p*. The Wasserstein distance *<sup>W</sup><sup>p</sup>* on *<sup>P</sup>*(R*<sup>d</sup>* ) is defined by *<sup>W</sup>p*(μ, ν) <sup>=</sup> *<sup>T</sup>p*(μ, ν) if *<sup>p</sup>* <sup>∈</sup> (0, <sup>1</sup>] and *<sup>W</sup>p*(μ, ν) <sup>=</sup> (*Tp*(μ, ν))1/*<sup>p</sup>* if *p* > 1.

The present paper studies the rate of convergence to zero of *Tp*(μ*<sup>N</sup>* , μ). This can be done in an asymptotic way, finding e.g. a sequence α(*N*) → 0 such that lim*<sup>N</sup>* α(*N*)−1*Tp*(μ*<sup>N</sup>* , μ) < <sup>∞</sup> a.s. or lim*<sup>N</sup>* α(*N*)−1E(*Tp*(μ*<sup>N</sup>* , μ)) < <sup>∞</sup>. Here we will rather derive some non-asymptotic moment estimates such as

$$\mathbb{E}(\mathcal{T}\_p(\mu\_N, \mu)) \le \alpha(N) \quad \text{for all } N \ge 1$$

as well as some non-asymptotic concentration estimates (also often called deviation inequalities)

$$\Pr(\mathcal{T}\_p(\mu\_N, \mu) \ge x) \le \alpha(N, x) \quad \text{for all } N \ge 1, \text{ all } x > 0.$$

They are naturally related to moment (or exponential moment) conditions on the law μ and we hope to derive an interesting interplay between the dimension *d* ≥ 1, the cost parameter *p* > 0 and these moment conditions. Let us introduce precisely these moment conditions. For *<sup>q</sup>* <sup>&</sup>gt; 0, α > 0, γ > 0 and <sup>μ</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ), we define

$$M\_q(\mu) := \int\_{\mathbb{R}^d} |\mathbf{x}|^q \mu(d\mathbf{x}) \quad \text{and} \quad \mathcal{E}\_{\alpha, \mathbf{y}}(\mu) := \int\_{\mathbb{R}^d} e^{\mathbf{y} \cdot |\mathbf{x}|^\alpha} \mu(d\mathbf{x}) .$$

We now present our main estimates, the comparison with the existing results and methods will be developped after this presentation. Let us however mention at once that our paper relies on some recent ideas of Dereich et al. [\[16](#page-30-0)].

#### 1.2 Moment estimates

<span id="page-1-0"></span>We first give some *L <sup>p</sup>* bounds.

**Theorem 1** *Let* <sup>μ</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ) *and let p* <sup>&</sup>gt; <sup>0</sup>*. Assume that Mq* (μ) < <sup>∞</sup> *for some q* <sup>&</sup>gt; *p. There exists a constant C depending only on p*, *d*, *q such that, for all N* ≥ 1*,*

$$\begin{aligned} \mathbb{E}\left(\mathcal{T}\_p(\mu\_N,\mu)\right) &\leq CM\_q^{p/q}(\mu) \\ \times \begin{cases} N^{-1/2} + N^{-(q-p)/q} & \text{if } p > d/2 \quad \text{and} \quad q \neq 2p, \\ N^{-1/2}\log(1+N) + N^{-(q-p)/q} & \text{if } p = d/2 \quad \text{and} \quad q \neq 2p, \\ N^{-p/d} + N^{-(q-p)/q} & \text{if } p \in (0, d/2) \quad \text{and} \quad q \neq d/(d-p). \end{cases} \end{aligned}$$

Observe that when μ has sufficiently many moments (namely if *q* > 2*p* when *<sup>p</sup>* <sup>≥</sup> *<sup>d</sup>*/2 and *<sup>q</sup>* <sup>&</sup>gt; *dp*/(*<sup>d</sup>* <sup>−</sup> *<sup>p</sup>*) when *<sup>p</sup>* <sup>∈</sup> (0, *<sup>d</sup>*/2)), the term *<sup>N</sup>*−(*q*−*p*)/*<sup>q</sup>* is small and can be removed. We could easily treat, for example, the case *p* > *d*/2 and *q* = 2*p* but this would lead to some logarithmic terms and the paper is technical enough.

This generalizes [\[16](#page-30-0)], in which only the case *p* ∈ [1, *d*/2) (whence *d* ≥ 3) and *q* > *dp*/(*d* − *p*) was treated. The argument is also slightly simplified.

To show that Theorem [1](#page-1-0) is really sharp, let us give examples where lower bounds can be derived quite precisely.

(a) If *<sup>a</sup>* = *<sup>b</sup>* <sup>∈</sup> <sup>R</sup>*<sup>d</sup>* and <sup>μ</sup> <sup>=</sup> (δ*<sup>a</sup>* <sup>+</sup>δ*b*)/2, one easily checks (see e.g. [\[16](#page-30-0), Remark 1]) thatE(*Tp*(μ*<sup>N</sup>* , μ)) <sup>≥</sup> *cN*−1/<sup>2</sup> for all *<sup>p</sup>* <sup>≥</sup> 1. Indeed, we haveμ*<sup>N</sup>* <sup>=</sup> *ZN* <sup>δ</sup>*a*+(1−*ZN* )δ*<sup>b</sup>* with *ZN* <sup>=</sup> *<sup>N</sup>*−<sup>1</sup> *<sup>N</sup>* <sup>1</sup> **<sup>1</sup>**{*Xi*=*a*}, so that *<sup>T</sup>p*(μ*<sup>N</sup>* , μ) = |*<sup>a</sup>* <sup>−</sup>*b*|*p*|*ZN* <sup>−</sup>1/2|, of which the expectation is of order *N*−1/2.

(b) Such a lower bound in *N*−1/<sup>2</sup> can easily be extended to any μ (possibly very smooth) of which the support is of the form *A* ∪ *B* with *d*(*A*, *B*) > 0 (simply note that *<sup>T</sup>p*(μ*<sup>N</sup>* , μ) <sup>≥</sup> *<sup>d</sup> <sup>p</sup>*(*A*, *<sup>B</sup>*)|*ZN* <sup>−</sup> μ(*A*)|, where *ZN* <sup>=</sup> *<sup>N</sup>*−<sup>1</sup> *<sup>N</sup>* <sup>1</sup> **1**{*Xi*∈*A*}).

(c) If μ is the uniform distribution on [−1, 1] *<sup>d</sup>* , it is well-known and not difficult to prove that for *<sup>p</sup>* <sup>&</sup>gt; 0, <sup>E</sup>(*Tp*(μ*<sup>N</sup>* , μ)) <sup>≥</sup> *cN*−*p*/*<sup>d</sup>* . Indeed, consider a partition of [−1, 1] *<sup>d</sup>* into (roughly) *N* cubes with length *N*−1/*<sup>d</sup>* . A quick computation shows that with probability greater than some *c* > 0 (uniformly in *N*), half of these cubes will not be charged by <sup>μ</sup>*<sup>N</sup>* . But on this event, we clearly have *<sup>T</sup>p*(μ*<sup>N</sup>* , μ) <sup>≥</sup> *aN*−1/*<sup>d</sup>* for some *a* > 0, because each time a cube is not charged by μ*<sup>N</sup>* , a (fixed) proportion of the mass of μ (in this cube) is at distance at least *N*−1/*<sup>d</sup>* /2 of the support of μ*<sup>N</sup>* . One easily concludes.

(d) When *p* = *d*/2 = 1, it has been shown by Ajtai et al. [\[2](#page-30-1)] that for μ the uniform measure on [−1, 1] *<sup>d</sup>* , *<sup>T</sup>*1(μ*<sup>N</sup>* , μ) *<sup>c</sup>*(log *<sup>N</sup>*/*N*)1/<sup>2</sup> with high probability, implying that <sup>E</sup>(*T*1(μ*<sup>N</sup>* , μ)) <sup>≥</sup> *<sup>c</sup>*(log *<sup>N</sup>*/*N*)1/2.

(e) Let μ(*dx*) = *c*|*x*| <sup>−</sup>*q*−*<sup>d</sup>* **<sup>1</sup>**{|*x*|≥1}*dx* for some *<sup>q</sup>* <sup>&</sup>gt; 0. Then *Mr*(μ) < <sup>∞</sup> for all *<sup>r</sup>* <sup>∈</sup> (0, *<sup>q</sup>*) and for all *<sup>p</sup>* <sup>≥</sup> 1, <sup>E</sup>(*Tp*(μ*<sup>N</sup>* , μ)) <sup>≥</sup> *cN*−(*q*−*p*)/*<sup>q</sup>* . Indeed, <sup>P</sup>(μ*<sup>N</sup>* ({|*x*| ≥ *<sup>N</sup>*1/*<sup>q</sup>* }) <sup>=</sup> <sup>0</sup>) <sup>=</sup> (μ({|*x*<sup>|</sup> <sup>&</sup>lt; *<sup>N</sup>*1/*<sup>q</sup>* }))*<sup>N</sup>* <sup>=</sup> (<sup>1</sup> <sup>−</sup> *<sup>c</sup>*/*N*)*<sup>N</sup>* <sup>≥</sup> *<sup>c</sup>* <sup>&</sup>gt; <sup>0</sup> and μ({|*x*| ≥ <sup>2</sup>*N*1/*<sup>q</sup>* }) <sup>≥</sup> *<sup>c</sup>*/*N*. One easily gets convinced that *<sup>T</sup>p*(μ*<sup>N</sup>* , μ) <sup>≥</sup> *<sup>N</sup> <sup>p</sup>*/*<sup>q</sup>* **<sup>1</sup>**{μ*<sup>N</sup>* ({|*x*|≥*N*1/*<sup>q</sup>* })=0}μ({|*x*| ≥ <sup>2</sup>*N*1/*<sup>q</sup>* }), from which the claim follows.

As far as *general laws* are concerned, Theorem [1](#page-1-0) is really sharp: the only possible improvements are the following. The first one, quite interesting, would be to replace log(1 + *N*) by something like log(1 + *N*) when *p* = *d*/2 (see point (d) above). It is however not clear whether it is feasible in full generality. The second one, which should be a mere (and not very interesting) refinement, would be to sharpen the bound in *<sup>N</sup>*−(*q*−*p*)/*<sup>q</sup>* when *Mq* (μ) < <sup>∞</sup>: point (e) only shows that there isμwith *Mq* (μ) < <sup>∞</sup> for which we have a lowerbound in *N*−(*q*−*p*)/*q*−<sup>ε</sup> for all ε > 0.

However, some improvements are possible when restricting the class of laws μ. First, when μ is the uniform distribution in [−1, 1] *<sup>d</sup>* , the results of Talagrand [\[38,](#page-31-2)[39\]](#page-31-3) strongly suggest that when *<sup>d</sup>* <sup>≥</sup> 3, <sup>E</sup>(*Tp*(μ*<sup>N</sup>* , μ)) *<sup>N</sup>*−*p*/*<sup>d</sup>* for all *<sup>p</sup>* <sup>&</sup>gt; 0, and this is much better than *N*−1/<sup>2</sup> when *p* is large. Such a result would of course immediately extend to any distribution <sup>μ</sup> <sup>=</sup> <sup>λ</sup> ◦ *<sup>F</sup>*−1, for <sup>λ</sup> the uniform distribution in [−1, <sup>1</sup>] *<sup>d</sup>* and *F* : [−1, 1] *<sup>d</sup>* <sup>→</sup> <sup>R</sup>*<sup>d</sup>* Lipschitz continuous. In any case, a smoothness assumption for μ cannot be sufficient, see point (b) above.

Second, for irregular laws, the convergence can be much faster than *N*−*p*/*<sup>d</sup>* when *p* < *d*/2, see point (a) above where, in an extreme case, we get *N*−1/<sup>2</sup> for all values of *p* > 0. It is shown by Dereich et al. [\[16\]](#page-30-0) (see also Barthe and Bordenave [\[3\]](#page-30-2)) that indeed, for a singular law, lim*<sup>N</sup> <sup>N</sup>*−*p*/*d*E(*Tp*(μ*<sup>N</sup>* , μ)) <sup>=</sup> 0.

# 1.3 Concentration inequalities

<span id="page-3-1"></span>We next state some concentration inequalities.

**Theorem 2** *Let* <sup>μ</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ) *and let p* <sup>&</sup>gt; <sup>0</sup>*. Assume one of the three following conditions:*

<span id="page-3-3"></span><span id="page-3-2"></span><span id="page-3-0"></span>
$$\exists \; \alpha > p, \; \exists \; \gamma > 0, \; \mathcal{E}\_{\alpha, \gamma}(\mu) < \infty,\tag{1}$$

$$\text{for } \exists \, \alpha \in (0, p), \,\, \exists \, \gamma > 0, \,\, \mathcal{E}\_{\alpha, \gamma}(\mu) < \infty,\tag{2}$$

$$\text{for} \quad \exists \neq q > 2p, \ M\_q(\mu) < \infty. \tag{3}$$

*Then for all N* ≥ 1*, all x* ∈ (0,∞)*,*

$$\mathbb{P}(\mathcal{T}\_p\left(\mu^N, \mu\right) \ge x) \le a(N, x)\mathbf{1}\_{\{x \le 1\}} + b(N, x),$$

*where*

$$a(N, \mathbf{x}) = C \begin{cases} \exp(-cN\mathbf{x}^2) & \text{if } p > d/2, \\ \exp(-cN(\mathbf{x}/\log(2 + 1/\mathbf{x}))^2) & \text{if } p = d/2, \\ \exp(-cN\mathbf{x}^{d/p}) & \text{if } p \in (0, d/2). \end{cases}$$

*and*

$$b(N, \mathbf{x}) = C \begin{cases} \exp(-cN\mathbf{x}^{\mathbf{a}/p})\mathbf{1}\_{\{\mathbf{x} \ge 1\}} & \text{under } (1), \\ \exp(-c(N\mathbf{x})^{(\mathbf{a}-\mathbf{c})/p})\mathbf{1}\_{\{\mathbf{x} \le 1\}} + \exp(-c(N\mathbf{x})^{\mathbf{a}/p})\mathbf{1}\_{\{\mathbf{x} \ge 1\}} & \forall \; \mathbf{c} \in (0, \mathbf{a}) \,\text{under } (2), \\ N(N\mathbf{x})^{-(q-\mathbf{c})/p} & \forall \; \mathbf{c} \in (0, q) \,\text{under } (3). \end{cases}$$

*The positive constants C and c depend only on p*, *d and either on* α, γ , *E*α,γ (μ)(*under* (1)) *or on* α, γ , *E*α,γ (μ), ε (*under* (2)) *or on q*, *Mq* (μ), ε (*under* (3))*.*

We could also treat the *critical* case where *E*α,γ (μ) < ∞ with α = *p*, but the result we could obtain is slightly more intricate and not very satisfying for small value of *x* (even if good for large ones).

*Remark 3* When assuming [\(2\)](#page-3-0) with α ∈ (0, *p*), we actually also prove that

$$b(N, \boldsymbol{x}) \le C \exp(-cN\boldsymbol{x}^2(\log(1+N))^{-\delta}) + C\exp(-c(N\boldsymbol{x})^{a/p}),$$

with δ = 2*p*/α − 1, see Step 5 of the proof of Lemma [13](#page-18-0) below. This allows us to extend the inequality *<sup>b</sup>*(*N*, *<sup>x</sup>*) <sup>≤</sup> *<sup>C</sup>* exp(−*c*(*N x*)α/*<sup>p</sup>*) to all values of *<sup>x</sup>* <sup>≥</sup> *xN* , for some (rather small) *xN* depending on *N*, α, *p*. But for very small values of *x* > 0, this formula is less interesting than that of Theorem [2.](#page-3-1) Despite much effort, we have not been able to get rid of the logarithmic term.

We believe that these estimates are quite satisfying. To get convinced, first observe that the *scales* seem to be the good ones. Recall that <sup>E</sup>(*Tp*(μ*<sup>N</sup>* , μ)) <sup>=</sup> <sup>∞</sup> <sup>0</sup> <sup>P</sup>(*Tp*(μ*<sup>N</sup>* , μ) <sup>≥</sup> *<sup>x</sup>*)*dx*.

(a) One easily checks that <sup>∞</sup> <sup>0</sup> *<sup>a</sup>*(*N*, *<sup>x</sup>*)*dx* <sup>≤</sup> *C N*−*p*/*<sup>d</sup>* if *<sup>p</sup>* <sup>&</sup>lt; *<sup>d</sup>*/2, *C N*−1/<sup>2</sup> log(1<sup>+</sup> *<sup>N</sup>*) if *<sup>p</sup>* <sup>=</sup> *<sup>d</sup>*/2, and *C N*−1/<sup>2</sup> if *<sup>p</sup>* <sup>&</sup>gt; *<sup>d</sup>*/2, as in Theorem [1.](#page-1-0)

(b) When integrating *<sup>b</sup>*(*N*, *<sup>x</sup>*) (or rather *<sup>b</sup>*(*N*, *<sup>x</sup>*) <sup>∧</sup> 1), we find *<sup>N</sup>*−(*q*−ε−*p*)/(*q*−ε) under [\(3\)](#page-3-2) and something smaller under (1) or (2). Since we can take *q* − ε > 2*p*, this is <sup>&</sup>lt; *<sup>N</sup>*−1/<sup>2</sup> (and thus also <sup>&</sup>lt; *<sup>N</sup>*−*p*/*<sup>d</sup>* if *<sup>p</sup>* <sup>&</sup>lt; *<sup>d</sup>*/2 and than *<sup>N</sup>*−1/<sup>2</sup> log(<sup>1</sup> <sup>+</sup> *<sup>N</sup>*) if *p* = *d*/2).

The *rates of decrease* are also satisfying in most cases. Recall that in deviation estimates, we never get something better than exp(−*N g*(*x*)) for some function *g*. Hence *a*(*N*, *x*) is probably optimal. Next, for *Y*¯*<sup>N</sup>* the empirical mean of a family of centered i.i.d. random variables, it is well-known that the *good* deviation inequalities are the following.

(a) If <sup>E</sup>[exp(*a*|*Y*1<sup>|</sup> β)] <sup>&</sup>lt; <sup>∞</sup> with <sup>β</sup> <sup>≥</sup> 1, then Pr[|*Y*¯*<sup>N</sup>* | ≥ *<sup>x</sup>*] ≤ *Ce*−*cNx*<sup>2</sup> **1**{*x*≤1} + *Ce*−*cNx*<sup>β</sup> **1**{*x*>1}, see for example Djellout et al. [\[18](#page-30-3)], Gozlan [\[24](#page-30-4)] or Ledoux [\[27](#page-30-5)], using transportation cost inequalities.

(b) If <sup>E</sup>[exp(*a*|*Y*1<sup>|</sup> β)] <sup>&</sup>lt; <sup>∞</sup> with β < 1, then Pr[|*Y*¯*<sup>N</sup>* | ≥ *<sup>x</sup>*] ≤ *Ce*−*cNx*<sup>2</sup> + *Ce*−*c*(*N x*)<sup>β</sup> , see Merlevède et al. [\[31,](#page-31-4) Formula (1.4)] which is based on results by Borovkov [\[8\]](#page-30-6).

(c) If <sup>E</sup>[|*Y*1<sup>|</sup> *<sup>r</sup>*] <sup>&</sup>lt; <sup>∞</sup> for some *<sup>r</sup>* <sup>&</sup>gt; 2, then Pr[|*Y*¯*<sup>N</sup>* | ≥ *<sup>x</sup>*] ≤ *Ce*−*cNx*<sup>2</sup> <sup>+</sup>*C N*(*N x*)−*r*, see Fuk and Nagaev [\[23\]](#page-30-7), using usual truncation arguments.

Our result is in perfect adequacy with these facts [(up to some arbitratry small loss due to ε under (2) and (3)] since *Tp*(μ*<sup>N</sup>* , μ) should behave very roughly as the mean of the <sup>|</sup>*Xi*|*p*'s, which e.g. has an exponential moment with power <sup>β</sup> := α/*<sup>p</sup>* under (1) and (2).

# 1.4 Comments

The control of the distance between the empirical measure of an i.i.d. sample and its true distribution is of course a long standing problem central both in probability, statistics and informatics with a wide number of applications: quantization (see Delattre et al. [\[14](#page-30-8)] and Pagès and Wilbertz [\[33\]](#page-31-5) for recent results), optimal matching (see Ajtai et al. [\[2](#page-30-1)], Dobri´c and Yukich [\[19](#page-30-9)], Talagrand [\[39](#page-31-3)], Barthe and Bordenave [\[3\]](#page-30-2)), density estimation, clustering (see Biau et al. [\[5\]](#page-30-10) and Laloë [\[26](#page-30-11)]), MCMC methods (see [\[36\]](#page-31-6) for bounds on ergodic averages), particle systems and approximations of partial differential equations (see Bolley et al. [\[11](#page-30-12)] and Fournier and Mischler [\[22](#page-30-13)]). We refer to these papers for an extensive introduction on this vast topic.

If many distances can be used to consider the problem, the Wasserstein distance is quite natural, in particular in quantization or for particle approximations of P.D.E.'s. However the depth of the problem was discovered only recently by Ajtai et al. [\[2](#page-30-1)], who considered the uniform measure on the square, investigated thoroughly by Talagrand [\[39](#page-31-3)]. As a review of the litterature is somewhat impossible, let us just say that the methods involved were focused on two methods inherited by the definitions of the Wasserstein distance: the construction of a coupling or by duality to control a particular empirical process.

Concerning moment estimates (as in Theorem [1\)](#page-1-0), some results can be found in Horowitz and Karandikar [\[25](#page-30-14)], Rachev and Rüschendorf [\[35\]](#page-31-7) and Mischler and Mouhot [\[32](#page-31-8)]. But theses results are far from optimal, even when assuming that μ is compactly supported. Very recently, strickingly clever alternatives were considered by Boissard and Le Gouic [\[7\]](#page-30-15) and by Dereich et al. [\[16](#page-30-0)]. Unfortunately, the construction of Boissard and Le Gouic, based on iterative trees, was a little too complicated to yield sharp rates. On the contrary, the method of [\[16](#page-30-0)], exposed in details in the next section, is extremely simple, robust, and leads to the almost optimal results exposed here. Some sharp moment estimates were already obtained in [\[16](#page-30-0)] for a limited range of parameters.

Concerning concentration estimates, only few results are available. Let us mention the work of Bolley et al. [\[11](#page-30-12)] and very recently by Boissard [\[6](#page-30-16)], which we considerably improve. Our assumptions are often much weaker (the reference measure μ was often assumed to satisfy some functional inequalities, which may be difficult to verify and usually include more "structure" than mere integrability conditions) and Pr[*Tp*(μ*<sup>N</sup>* , μ) ≥ *x*] was estimated only for rather large values of *x*. In particular, when integrating the concentration estimates of [\[11\]](#page-30-12), one does never find the *good* moment estimates, meaning that the scales are not the good ones.

Moreover, the approach of [\[16](#page-30-0)] is robust enough so that we can also give some good moment bounds for the Wasserstein distance between the empirical measure of a Markov chain and its invariant distribution (under some conditions). This could be useful for MCMC methods because our results are non asymptotic. We can also study very easily some ρ-mixing sequences (see Doukhan [\[20\]](#page-30-17)), for which only very few results exist, see Biau et al. [\[7\]](#page-30-15). Finally, we show on an example how to use Theorem [1](#page-1-0) to study some particle systems. For all these problems, we might also obtain some concentration inequalities, but this would need further refinements which are out of the scope of the present paper, somewhat already technical enough, and left for further works.

#### 1.5 Plan of the paper

In the next section, we state some general upper bounds of *Tp*(μ, ν), for any μ, ν ∈ *<sup>P</sup>*(R*<sup>d</sup>* ), essentially taken from [\[16](#page-30-0)]. Section [3](#page-9-0) is devoted to the proof of Theorem [1.](#page-1-0) Theorem [2](#page-3-1) is proved in three steps: in Sect. [4](#page-11-0) we study the case where μ is compactly supported and where *N* is replaced by a Poisson(*N*)-distributed random variable, which yields some pleasant independance properties. We show how to remove the randomization in Sect. [5,](#page-15-0) concluding the case where μ is compactly supported. The non compact case is studied in Sect. [6.](#page-17-0) The final Sect. [7](#page-25-0) is devoted to dependent random variables: ρ-mixing sequences, Markov chains and a particular particle system.

### **2 Coupling**

<span id="page-6-1"></span>The following notion of distance, essentially taken from [\[16](#page-30-0)], is the main ingredient of the paper.

**Notation 4** (*a*) *For* ≥ 0*, we denote by P the natural partition of* (−1, 1] *<sup>d</sup> into* 2*<sup>d</sup> translations of* (−2−, <sup>2</sup><sup>−</sup>] *<sup>d</sup> . For two probability measures* μ, ν *on* (−1, <sup>1</sup>] *<sup>d</sup> and for p* > 0*, we introduce*

$$\mathcal{D}\_p(\mu, \nu) := \frac{2^p - 1}{2} \sum\_{\ell \ge 1} 2^{-p\ell} \sum\_{F \in \mathcal{P}\_\ell} |\mu(F) - \nu(F)|,$$

*which obviously defines a distance on P*((−1, 1] *<sup>d</sup>* )*, always bounded by* 1*.*

(*b*) *We introduce B*<sup>0</sup> := (−1, 1] *<sup>d</sup> and, for n* <sup>≥</sup> <sup>1</sup>*, Bn* := (−2*n*, <sup>2</sup>*n*] *<sup>d</sup>* \(−2*n*−1, <sup>2</sup>*n*−1] *d . For* <sup>μ</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ) *and n* <sup>≥</sup> <sup>0</sup>*, we denote by <sup>R</sup>Bn*<sup>μ</sup> *the probability measure on* (−1, <sup>1</sup>] *d defined as the image of* <sup>μ</sup>|*Bn* /μ(*Bn*) *by the map x* <sup>→</sup> *<sup>x</sup>*/2*n. For two probability measures* μ, ν *on* R*<sup>d</sup> and for p* > 0*, we introduce*

$$\mathcal{D}\_p(\mu, \boldsymbol{\nu}) := \sum\_{n \ge 0} 2^{pn} \left( |\mu(B\_n) - \boldsymbol{\nu}(B\_n)| + (\mu(B\_n) \wedge \boldsymbol{\nu}(B\_n)) \mathcal{D}\_p(\mathcal{R}\_{B\_n} \boldsymbol{\mu}, \mathcal{R}\_{B\_n} \boldsymbol{\nu}) \right).$$

*A little study, using that D<sup>p</sup>* ≤ 1 *on P*((−1, 1] *<sup>d</sup>* )*, shows that this defines a distance on <sup>P</sup>*(R*<sup>d</sup>* )*.*

Having a look at *D<sup>p</sup>* in the compact case, one sees that in some sense, it measures distance of the two probability measures simultaneously at all the scales. The optimization procedure can be made for all scales and outperforms the approach based on a fixed diameter covering of the state space (which is more or less the approach of Horowitz and Karandikar [\[25\]](#page-30-14)). Moreover one sees that the principal control is on |π(*F*)−μ(*F*)| which is a quite simple quantity. The next results are slightly modified versions of estimates found in [\[16\]](#page-30-0), see [\[16](#page-30-0), Lemma 2] for the compact case and [\[16,](#page-30-0) proof of Theorem 3] for the non compact case. It contains the crucial remark that *D<sup>p</sup>* is an upper bound (up to constant) of the Wasserstein distance.

<span id="page-6-0"></span>**Lemma 5** *Let d* <sup>≥</sup> <sup>1</sup> *and p* <sup>&</sup>gt; <sup>0</sup>*. For all pairs of probability measures* μ, ν *on* <sup>R</sup>*<sup>d</sup> , <sup>T</sup>p*(μ, ν) <sup>≤</sup> <sup>κ</sup>*p*,*dDp*(μ, ν)*, with* <sup>κ</sup>*p*,*<sup>d</sup>* := <sup>2</sup>*pd <sup>p</sup>*/2(2*<sup>p</sup>* <sup>+</sup> <sup>1</sup>)/(2*<sup>p</sup>* <sup>−</sup> <sup>1</sup>)*.*

*Proof* We separate the proof into two steps.

**Step 1.** We first assume that μ and ν are supported in (−1, 1] *<sup>d</sup>* . We infer from [\[16,](#page-30-0) Lemma 2], in which the conditions *p* ≥ 1 and *d* ≥ 3 are clearly not used, that, since the diameter of (−1, 1] *<sup>d</sup>* is 2√*d*,

$$\mathcal{T}\_p(\mu, \upsilon) \le \frac{\left(2\sqrt{d}\right)^p}{2} \sum\_{\ell \ge 0} 2^{-p\ell} \sum\_{F \in \mathcal{P}\_\ell} \mu(F) \sum\_{\substack{C \text{ child of } F}} \left| \frac{\mu(C)}{\mu(F)} - \frac{\nu(C)}{\nu(F)} \right|,$$

where "*C* child of *F*" means that *C* ∈ *P*<sup>+</sup><sup>1</sup> and *C* ⊂ *F*. Consequently,

$$\begin{split} T\_{\mathcal{P}}(\mu,\nu) &\leq 2^{p-1}d^{p/2} \sum\_{\ell\geq 0} 2^{-p\ell} \sum\_{F\in\mathcal{P}\_{\ell}} \sum\_{\begin{subarray}{c} C\text{ child of }\operatorname{\boldsymbol{\mathcal{F}}}\text{ } \end{subarray}} \left( \frac{\operatorname{\boldsymbol{\nu}}(C)}{\operatorname{\boldsymbol{\nu}}(F)} |\mu(F) - \boldsymbol{\nu}(F)| + |\mu(C) - \boldsymbol{\nu}(C)| \right) \\ &\leq 2^{p-1}d^{p/2} \sum\_{\ell\geq 0} 2^{-p\ell} \left( \sum\_{F\in\mathcal{P}\_{\ell}} |\mu(F) - \boldsymbol{\nu}(F)| + \sum\_{C\in\mathcal{P}\_{\ell+1}} |\mu(C) - \boldsymbol{\nu}(C)| \right) \\ &\leq 2^{p-1}d^{p/2} (1 + 2^p) \sum\_{\ell\geq 1} 2^{-p\ell} \sum\_{F\in\mathcal{P}\_{\ell}} |\mu(F) - \boldsymbol{\nu}(F)|, \end{split}$$

which is nothing but κ*p*,*dDp*(μ, ν). We used that *<sup>F</sup>*∈*P*<sup>0</sup> <sup>|</sup>μ(*F*) <sup>−</sup> ν(*F*)| = 0.

In Dereich et al. [\[16\]](#page-30-0), use directly the formula with the children to study the rate of convergence of empirical measures. This leads to some (small) technical complications, and does not seem to improve the estimates.

**Step 2.** We next consider the general case. We consider, for each *n* ≥ 1, the optimal coupling π*n*(*dx*, *dy*) between *RBn*μ and *RBn* ν for *Tp*. We define ξ*n*(*dx*, *dy*) as the image of <sup>π</sup>*<sup>n</sup>* by the map (*x*, *<sup>y</sup>*) <sup>→</sup> (2*<sup>n</sup> <sup>x</sup>*, <sup>2</sup>*<sup>n</sup> <sup>y</sup>*), which clearly belongs to *<sup>H</sup>*(μ|*Bn* /μ(*Bn*), ν|*Bn* /ν(*Bn*)) and satisfies <sup>|</sup>*<sup>x</sup>* <sup>−</sup> *<sup>y</sup>*|*p*ξ*n*(*dx*, *dy*) <sup>=</sup> <sup>2</sup>*np* <sup>|</sup>*<sup>x</sup>* <sup>−</sup> *<sup>y</sup>*|*p*π*n*(*dx*, *dy*) <sup>=</sup> <sup>2</sup>*npTp*(*RBn*μ, *<sup>R</sup>Bn* ν).

Next, we introduce *<sup>q</sup>* := <sup>1</sup> 2 *<sup>n</sup>*≥<sup>0</sup> <sup>|</sup>ν(*Bn*) <sup>−</sup> μ(*Bn*)<sup>|</sup> and we define

$$
\xi(d\chi, d\text{y}) = \sum\_{n\geq 0} (\mu(B\_n) \wedge \nu(B\_n)) \xi\_n(d\chi, d\text{y}) + \frac{\alpha(d\text{x})\beta(d\text{y})}{q},
$$

where

$$\alpha(d\boldsymbol{\omega}) := \sum\_{n\geq 0} (\mu(B\_n) - \upsilon(B\_n)) + \frac{\mu|\_{B\_n}(d\boldsymbol{\omega})}{\mu(B\_n)} \quad \text{and}$$

$$\beta(d\boldsymbol{\upy}) := \sum\_{n\geq 0} (\upsilon(B\_n) - \mu(B\_n)) + \frac{\upsilon|\_{B\_n}(d\boldsymbol{\upy})}{\upsilon(B\_n)}.$$

Using that

$$\begin{aligned} q &= \sum\_{n\geq 0} (\upsilon(B\_n) - \mu(B\_n))\_+ = \sum\_{n\geq 0} (\mu(B\_n) - \upsilon(B\_n))\_+ \\ &= 1 - \sum\_{n\geq 0} (\upsilon(B\_n) \wedge \mu(B\_n)), \end{aligned}$$

it is easily checked that ξ ∈ *H*(μ, ν). Furthermore, we have, setting *cp* = 1 if *<sup>p</sup>* <sup>∈</sup> (0, <sup>1</sup>] and *cp* <sup>=</sup> <sup>2</sup>*p*−<sup>1</sup> if *<sup>p</sup>* <sup>&</sup>gt; 1,

$$\begin{split} \iint |\mathbf{x} - \mathbf{y}|^{p} \frac{\alpha(d\mathbf{x})\beta(d\mathbf{y})}{q} &\leq \frac{1}{q} \iint c\_{p}(|\mathbf{x}|^{p} + |\mathbf{y}|^{p})\alpha(d\mathbf{x})\beta(d\mathbf{y}) \\ &= c\_{p} \int |\mathbf{x}|^{p}\alpha(d\mathbf{x}) + c\_{p} \int |\mathbf{y}|^{p}\beta(d\mathbf{y}) \\ &\leq c\_{p} \sum\_{n\geq 0} 2^{pn} [(\mu(B\_{n}) - \nu(B\_{n}))\_{+} + (\nu(B\_{n}) - \mu(B\_{n}))\_{+}] \\ &= c\_{p} \sum\_{n\geq 0} 2^{pn} |\mu(B\_{n}) - \nu(B\_{n})|. \end{split}$$

Recalling that <sup>|</sup>*<sup>x</sup>* <sup>−</sup> *<sup>y</sup>*|*p*ξ*n*(*dx*, *dy*) <sup>≤</sup> <sup>2</sup>*npTp*(*RBn*μ, *<sup>R</sup>Bn* ν), we deduce that

$$\begin{aligned} \left| \mathcal{T}\_p(\mu, \boldsymbol{\nu}) \right| &\leq \int |\boldsymbol{\chi} - \boldsymbol{\chi}|^p \xi(d\boldsymbol{x}, d\boldsymbol{y}) \\ &\leq \sum\_{n\geq 0} 2^{np} \left( c\_p |\mu(B\_n) - \boldsymbol{\nu}(B\_n)| \\ &\quad + (\mu(B\_n) \wedge \boldsymbol{\nu}(B\_n)) \mathcal{T}\_p(\mathcal{R}\_{B\_n} \boldsymbol{\mu}, \mathcal{R}\_{B\_n} \boldsymbol{\nu}) \right) \end{aligned}$$

We conclude using Step 1 and that *cp* ≤ κ*p*,*<sup>d</sup>* .

When proving the concentration inequalities, which is very technical, it will be good to break the proof into several steps to separate the difficulties and we will first treat the compact case. On the contrary, when dealing with moment estimates, the following formula will be easier to work with.

<span id="page-8-0"></span>**Lemma 6** *Let p* <sup>&</sup>gt; <sup>0</sup> *and d* <sup>≥</sup> <sup>1</sup>*. For all* μ, ν <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* )*,*

$$\mathcal{D}\_p(\mu, \nu) \le C\_p \sum\_{n \ge 0} 2^{pn} \sum\_{\ell \ge 0} 2^{-p\ell} \sum\_{F \in \mathcal{P}\_\ell} \left| \mu(2^n F \cap B\_n) - \nu(2^n F \cap B\_n) \right|.$$

*with the notation* <sup>2</sup>*<sup>n</sup> <sup>F</sup>* = {2*<sup>n</sup> <sup>x</sup>* : *<sup>x</sup>* <sup>∈</sup> *<sup>F</sup>*} *and where Cp* <sup>=</sup> <sup>1</sup> <sup>+</sup> <sup>2</sup>−*p*/(<sup>1</sup> <sup>−</sup> <sup>2</sup>−*p*)*.*

*Proof* For all *n* ≥ 1, we have |μ(*Bn*) − ν(*Bn*)| = *<sup>F</sup>*∈*P*<sup>0</sup> <sup>|</sup>μ(2*<sup>n</sup> <sup>F</sup>* <sup>∩</sup> *Bn*) <sup>−</sup> ν(2*<sup>n</sup> <sup>F</sup>*<sup>∩</sup> *Bn*)| and

$$\begin{split} & (\mu(B\_n) \wedge \upsilon(B\_n)) \mathcal{D}\_p(\mathcal{R}\_{B\_n} \mu, \mathcal{R}\_{B\_n} \upsilon) \\ & \leq \mu(B\_n) \sum\_{\ell \geq 1} 2^{-p\ell} \sum\_{F \in \mathcal{P}\_\ell} \left| \frac{\mu(2^n F \cap B\_n)}{\mu(B\_n)} - \frac{\upsilon(2^n F \cap B\_n)}{\upsilon(B\_n)} \right| \\ & \leq \sum\_{\ell \geq 1} 2^{-p\ell} \sum\_{F \in \mathcal{P}\_\ell} \left| \mu(2^n F \cap B\_n) - \upsilon(2^n F \cap B\_n) \right| \\ & + \left| 1 - \frac{\mu(B\_n)}{\upsilon(B\_n)} \right| \sum\_{\ell \geq 1} 2^{-p\ell} \sum\_{F \in \mathcal{P}\_\ell} \upsilon(2^n F \cap B\_n). \end{split}$$

This last term is smaller than 2−*<sup>p</sup>* <sup>|</sup>μ(*Bn*) <sup>−</sup> ν(*Bn*)<sup>|</sup> /(1−2−*p*) and this ends the proof. 

123

.

### <span id="page-9-0"></span>**3 Moment estimates**

The aim of this section is to give the

*Proof of Theorem 1* We thus assume that <sup>μ</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ) and that *Mq* (μ) < <sup>∞</sup> for some *q* > *p*. By a scaling argument, we may assume that *Mq* (μ) = 1. This implies that μ(*Bn*) <sup>≤</sup> <sup>2</sup>−*q*(*n*−1) for all *<sup>n</sup>* <sup>≥</sup> 0. By Lemma [5,](#page-6-0) we have *<sup>T</sup>p*(μ*<sup>N</sup>* , μ) <sup>≤</sup> <sup>κ</sup>*p*,*dDp*(μ*<sup>N</sup>* , μ), so that it suffices to study <sup>E</sup>(*Dp*(μ*<sup>N</sup>* , μ)). In the whole proof, the positive constant *<sup>C</sup>*, whose value may change from line to line, depends only on *p*, *d*, *q*.

For a Borel subset *<sup>A</sup>* <sup>⊂</sup> <sup>R</sup>*<sup>d</sup>* , since *<sup>N</sup>*μ*<sup>N</sup>* (*A*) is Binomial(*N*, μ(*A*))-distributed, we have

$$\mathbb{E}\left(|\mu\_N(A) - \mu(A)|\right) \le \min\left\{2\mu(A), \sqrt{\mu(A)/N}\right\}.$$

Using the Cauchy–Scharz inequality and that #(*P*) <sup>=</sup> <sup>2</sup>*<sup>d</sup>*, we deduce that for all *n* ≥ 0, all ≥ 0,

$$\sum\_{F \in \mathcal{P}\_{\ell}} \mathbb{E}\left( \left| \mu\_N(\mathcal{Z}^\mathbb{R} F \cap B\_n) - \mu(\mathcal{Z}^\mathbb{R} F \cap B\_n) \right| \right) \le \min \left\{ 2\mu(B\_n), \mathcal{Z}^{d\ell/2} (\mu(B\_n)/N)^{1/2} \right\}.$$

Using finally Lemma [6](#page-8-0) and that μ(*Bn*) <sup>≤</sup> <sup>2</sup>−*q*(*n*−1) , we find

<span id="page-9-1"></span>
$$\mathbb{E}(\mathcal{D}\_p(\mu\_N, \mu)) \le C \sum\_{n\ge 0} 2^{pn} \sum\_{\ell \ge 0} 2^{-p\ell} \min\left\{ 2^{-qn}, 2^{d\ell/2} (2^{-qn}/N)^{1/2} \right\}.\tag{4}$$

**Step 1.** Here we show that for all ε ∈ (0, 1), all *N* ≥ 1,

$$\sum\_{\ell \ge 0} 2^{-p\ell} \min \left\{ \varepsilon, 2^{d\ell/2} (\varepsilon/N)^{1/2} \right\} \le C \begin{cases} \min \{ \varepsilon, (\varepsilon/N)^{1/2} \} & \text{if } p > d/2, \\ \min \{ \varepsilon, (\varepsilon/N)^{1/2} \log(2 + \varepsilon N) \} & \text{if } p = d/2, \\ \min \{ \varepsilon, \varepsilon (\varepsilon N)^{-p/d} \} & \text{if } p \in (0, d/2). \end{cases}$$

First of all, the bound by *C*ε is obvious in all cases (because *p* > 0). Next, the case *p* > *d*/2 is immediate. If *p* ≤ *d*/2, we introduce *<sup>N</sup>*,ε := log(2 + ε*N*)/(*d* log 2), for which 2*<sup>d</sup><sup>N</sup>*,ε <sup>2</sup> <sup>+</sup> <sup>ε</sup>*<sup>N</sup>* and get an upper bound in

$$(\mathfrak{s}/N)^{1/2} \sum\_{\ell \le \ell\_{N,\mathfrak{s}}} 2^{(d/2-p)\ell} + \mathfrak{s} \sum\_{\ell \ge \ell\_{N,\mathfrak{s}}} 2^{-p\ell} \cdot \mathfrak{s}$$

If *p* = *d*/2, we find an upper bound in

$$\begin{aligned} (\varepsilon/N)^{1/2} \ell\_{N,\varepsilon} + C\varepsilon 2^{-p\ell\_{N,\varepsilon}} &\leq C(\varepsilon/N)^{1/2} \log(2+\varepsilon N) \\ + C\varepsilon (1+\varepsilon N)^{-1/2} &\leq C(\varepsilon/N)^{1/2} \log(2+\varepsilon N) \end{aligned}$$

as desired. If *p* ∈ (0, *d*/2), we get an upper bound in

$$C(\varepsilon/N)^{1/2}2^{(d/2-p)\ell\_{N,\varepsilon}} + C\varepsilon 2^{-p\ell\_{N,\varepsilon}} \le C(\varepsilon/N)^{1/2}(2+\varepsilon N)^{1/2-p/d} + C\varepsilon(2+\varepsilon N)^{-p/d}.$$

If <sup>ε</sup>*<sup>N</sup>* <sup>≥</sup> 1, then (2+ε*N*)1/2−*p*/*<sup>d</sup>* <sup>≤</sup> (3ε*N*)1/2−*p*/*<sup>d</sup>* and the conclusion follows. If now <sup>ε</sup>*<sup>N</sup>* <sup>∈</sup> (0, <sup>1</sup>), the result is obvious because min{ε, ε(ε*N*)−*p*/*<sup>d</sup>* } = <sup>ε</sup>.

**Step 2:** *<sup>p</sup>* <sup>&</sup>gt; *<sup>d</sup>*/2. By [\(4\)](#page-9-1) and Step 1 (with <sup>ε</sup> <sup>=</sup> <sup>2</sup>−*qn*), we find

$$\begin{aligned} \mathbb{E}(\mathcal{D}\_p(\mu\_N, \mu)) &\leq C \sum\_{n\geq 0} 2^{pn} \min\left\{ 2^{-qn}, (2^{-qn}/N)^{1/2} \right\} \\ &\leq C \begin{cases} N^{-1/2} & \text{if } q > 2p, \\ N^{-(q-p)/q} & \text{if } q \in (p, 2p). \end{cases} \end{aligned}$$

Indeed, this is obvious if *q* > 2*p*, while the case *q* ∈ (*p*, 2*p*) requires to separate the sum in two parts *n* ≤ *nN* and *n* > *nN* with *nN* = log *N*/(*q* log 2). This ends the proof when *p* > *d*/2.

**Step 3:** *<sup>p</sup>* <sup>=</sup> *<sup>d</sup>*/2. By [\(4\)](#page-9-1) and Step 1 (with <sup>ε</sup> <sup>=</sup> <sup>2</sup>−*qn*), we find

$$\mathbb{E}(\mathcal{D}\_p(\mu\_N, \mu)) \le C \sum\_{n \ge 0} 2^{pn} \min \left\{ 2^{-qn}, (2^{-qn}/N)^{1/2} \log(2 + 2^{-qn}N) \right\}.$$

If *q* > 2*p*, we immediately get a bound in

$$\mathbb{E}(\mathcal{D}\_p(\mu\_N, \mu)) \le C \sum\_{n \ge 0} 2^{(p - q/2)n} N^{-1/2} \log(2 + N) \le C \log(2 + N) N^{-1/2},$$

which ends the proof (when *p* = *d*/2 and *q* > 2*p*).

If *q* ∈ (*p*, 2*p*), we easily obtain, using that log(2 + *x*) ≤ 2 log *x* for all *x* ≥ 2, an upper bound in

$$\begin{split} \mathbb{E}(\mathcal{D}\_{p}(\mu\_{N},\mu)) &\leq C \sum\_{n\geq 0} \mathbf{1}\_{\{N < 2.2^{nq}\}} 2^{(p-q)n} + C \sum\_{n\geq 0} \mathbf{1}\_{\{N \geq 2.2^{nq}\}} 2^{(p-q/2)n} N^{-1/2} \log(N 2^{-nq}) \\ &\leq C N^{-(q-p)/q} + C N^{-1/2} \sum\_{n=0}^{n\_N} 2^{(p-q/2)n} (\log N - nq \log 2) \\ &=: C N^{-(q-p)/q} + C N^{-1/2} K\_N, \end{split}$$

where *nN* = log(*N*/2)/(*q* log 2). A tedious exact computation shows that

$$\begin{aligned} K\_N &= \log N \frac{2^{(p-q/2)(n\_N+1)} - 1}{2^{(p-q/2)} - 1} \\ &- q \log 2 \left[ (n\_N+1) \frac{2^{(p-q/2)(n\_N+1)} - 1}{2^{(p-q/2)} - 1} + \frac{n\_N+1}{2^{(p-q/2)} - 1} \right] \\ &- \frac{2^{(p-q/2)(n\_N+2)} - 2^{(p-q/2)}}{(2^{(p-q/2)} - 1)^2} \end{aligned}$$

Using that the contribution of the middle term of the second line is negative and the inequality log *N* − (*nN* + 1)*q* log 2 ≤ log 2 (because (*nN* + 1)*q* log 2 ≥ log(*N*/2)), we find

$$K\_N \le C 2^{(p-q/2)n\_N} \le C N^{p/q - 1/2}.$$

We finally have checked that <sup>E</sup>(*Dp*(μ*<sup>N</sup>* , μ)) <sup>≤</sup> *C N*−(*q*−*p*)/*<sup>q</sup>* <sup>+</sup> *C N*−1/2*<sup>N</sup> <sup>p</sup>*/*q*−1/<sup>2</sup> <sup>≤</sup> *C N*−(*q*−*p*)/*<sup>q</sup>* , which ends the proof when *<sup>p</sup>* <sup>=</sup> *<sup>d</sup>*/2.

**Step 4:** *p* ∈ (0, *d*/2). We then have, by [\(4\)](#page-9-1) and Step 1,

$$\mathbb{E}\left(\mathcal{D}\_p(\mu\_N,\mu)\right) \le C \sum\_{n\ge 0} 2^{pn} \min\left\{ 2^{-qn}, 2^{-qn(1-p/d)}N^{-p/d} \right\}.$$

If *q* > *dp*/(*d* − *p*), which implies that *q*(1 − *p*/*d*) > *p*, we immediately get an upper bound by *C N*−*p*/*<sup>d</sup>* , which ends the proof when *<sup>p</sup>* <sup>&</sup>lt; *<sup>d</sup>*/2 and *<sup>q</sup>* <sup>&</sup>gt; *dp*/(*<sup>d</sup>* <sup>−</sup> *<sup>p</sup>*).

If finally *q* ∈ (*p*, *dp*/(*d* − *p*)), we separate the sum in two parts *n* ≤ *nN* and *<sup>n</sup>* <sup>&</sup>gt; *nN* with *nN* = log *<sup>N</sup>*/(*<sup>q</sup>* log 2) and we find a bound in *C N*−(*q*−*p*)/*<sup>q</sup>* as desired. 

#### <span id="page-11-0"></span>**4 Concentration inequalities in the compact poissonized case**

It is technically advantageous to first consider the case where the size of the sampling is Poisson distributed, which implies some independence properties. Replacing *N* (large) by a Poisson(*N*)-distributed random variable should be feasible, because a Poisson(*N*)-distributed random variable is close to *N* with high probability.

<span id="page-11-2"></span>**Notation 7** *We introduce the functions f and g defined on* (0,∞) *by*

$$f(\mathbf{x}) = (1+\mathbf{x})\log(1+\mathbf{x}) - \mathbf{x} \quad \text{and} \quad \mathbf{g}(\mathbf{x}) = (\mathbf{x}\log\mathbf{x} - \mathbf{x} + 1)\mathbf{1}\_{\{\mathbf{x} \ge \mathbf{l}\}}.$$

<span id="page-11-1"></span>*Observe that f is increasing, nonnegative, equivalent to x*<sup>2</sup> *at* 0 *and to x* log *x at infinity. The function g is positive and increasing on* (1,∞)*.*

The goal of this section is to check the following.

**Proposition 8** *Assume that* μ *is supported in* (−1, 1] *<sup>d</sup> . Let <sup>N</sup> be a Poisson measure on* R*<sup>d</sup> with intensity measure N*μ *and introduce the associated empirical measure <sup>N</sup>* <sup>=</sup> (*<sup>N</sup>* (R*<sup>d</sup>* ))−1*<sup>N</sup> . Let p* <sup>≥</sup> <sup>1</sup> *and d* <sup>≥</sup> <sup>1</sup>*. There are some positive constants C*, *c* (*depending only on d*, *p*) *such that for all N* ≥ 1*, all x* ∈ (0,∞)*,*

$$\mathbb{P}\left(\Pi\_{N}(\mathbb{R}^{d})\mathcal{D}\_{p}(\Psi\_{N},\mu)\geq Nx\right)\leq C\begin{cases} \exp(-Nf(c\mathbf{x})) & \text{if } p>d/2,\\\exp\left(-Nf(c\mathbf{x}/\log(2+\mathbf{l}/\mathbf{x}))\right) & \text{if } p=d/2,\\\exp\left(-Nf(c\mathbf{x})\right)+\exp\left(-cNx^{d/p}\right) & \text{if } p\in(0,d/2). \end{cases}$$

<span id="page-12-0"></span>We start with some easy and well-known concentration inequalities for the Poisson distribution.

**Lemma 9** *For* λ > 0 *and X a Poisson*(λ)*-distributed random variable, we have*

(a) *<sup>E</sup>*(exp(θ *<sup>X</sup>*)) <sup>=</sup> exp(λ(*e*<sup>θ</sup> <sup>−</sup> <sup>1</sup>)) *for all* <sup>θ</sup> <sup>∈</sup> <sup>R</sup>*;* (b) *<sup>E</sup>*(exp(θ|*<sup>X</sup>* <sup>−</sup> <sup>λ</sup>|)) <sup>≤</sup> 2 exp(λ(*e*<sup>θ</sup> <sup>−</sup> <sup>1</sup> <sup>−</sup> θ )) *for all* θ > <sup>0</sup>*;* (c) <sup>P</sup>(*<sup>X</sup>* > λ*x*) <sup>≤</sup> exp(−λ*g*(*x*)) *for all x* <sup>&</sup>gt; <sup>0</sup>*;* (d) <sup>P</sup>(|*<sup>X</sup>* <sup>−</sup> <sup>λ</sup><sup>|</sup> > λ*x*) <sup>≤</sup> 2 exp(−λ*<sup>f</sup>* (*x*)) *for all x* <sup>&</sup>gt; <sup>0</sup>*;* (e) <sup>P</sup>(*<sup>X</sup>* > λ*x*) <sup>≤</sup> <sup>λ</sup> *for all x* <sup>&</sup>gt; <sup>0</sup>*.*

*Proof* Point (a) is straightforward. For point (b), write *E*(exp(θ|*X* − λ|)) ≤ *<sup>e</sup>*θλE(exp(−<sup>θ</sup> *<sup>X</sup>*))+*e*−θλE(exp(θ *<sup>X</sup>*)), use (a) and that λ(*e*−<sup>θ</sup> <sup>−</sup>1+θ ) <sup>≤</sup> λ(*e*<sup>θ</sup> <sup>−</sup>1−θ ). For point (c), write <sup>P</sup>(*<sup>X</sup>* > λ*x*) <sup>≤</sup> *<sup>e</sup>*−θλ*x*E[exp(θ *<sup>X</sup>*)], use (a) and optimize in <sup>θ</sup>. Use the same scheme to deduce (d) from (b). Finally, for *<sup>x</sup>* <sup>&</sup>gt; 0, <sup>P</sup>(*<sup>X</sup>* > λ*x*) <sup>≤</sup> <sup>P</sup>(*<sup>X</sup>* <sup>&</sup>gt; <sup>0</sup>) <sup>=</sup> <sup>1</sup> <sup>−</sup> *<sup>e</sup>*−<sup>λ</sup> <sup>≤</sup> <sup>λ</sup>.

We can now give the

*Proof of Proposition 8* During the proof, the constants may only depend on *p* and *d*. We fix *x* > 0 for the whole proof. Recalling Notation [4-](#page-6-1)(a), we have

$$\begin{split} \Pi\_{N}(\mathbb{R}^{d})\mathcal{D}\_{p}(\Psi\_{N},\mu) &= C\sum\_{\ell\geq 1}2^{-p\ell}\sum\_{F\in\mathcal{P}\_{\ell}}|\Pi\_{N}(F)-\Pi\_{N}(\mathbb{R}^{d})\mu(F)| \\ &\leq C|\Pi\_{N}(\mathbb{R}^{d})-N| + C\sum\_{\ell\geq 1}2^{-p\ell}\sum\_{F\in\mathcal{P}\_{\ell}}|\Pi\_{N}(F)-N\mu(F)| \\ &\leq C|\Pi\_{N}(\mathbb{R}^{d})-N| + C(N+\Pi\_{N}(\mathbb{R}^{d}))2^{-p\ell\_{0}} \\ &+ C\sum\_{\ell=1}^{\ell\_{0}}2^{-p\ell}\sum\_{F\in\mathcal{P}\_{\ell}}|\Pi\_{N}(F)-N\mu(F)| \end{split}$$

for any choice of <sup>0</sup> <sup>∈</sup> <sup>N</sup>. We will choose <sup>0</sup> later, depending on the value of *<sup>x</sup>*. For any nonnegative family *r* such that <sup>0</sup> <sup>1</sup> *r* ≤ 1, we thus have

$$\begin{aligned} \varepsilon(N, \boldsymbol{x}) &:= \mathbb{P}\left(\boldsymbol{\Pi}\_N(\mathbb{R}^d) \mathcal{D}\_p(\boldsymbol{\Psi}\_N, \boldsymbol{\mu}) \ge N\boldsymbol{x}\right) \\ &\le \mathbb{P}\left(|\boldsymbol{\Pi}\_N(\mathbb{R}^d) - N| \ge cN\boldsymbol{x}\right) + \mathbb{P}\left(\boldsymbol{\Pi}\_N(\mathbb{R}^d) \ge N(c\boldsymbol{x}2^{p\ell\_0} - 1)\right) \end{aligned}$$

$$+\sum\_{\ell=1}^{\ell\_0} \mathbb{P}\left(\sum\_{F \in \mathcal{P}\_\ell} |\Pi\_N(F) - N\mu(F)| \ge cNx2^{p\ell}r\_\ell\right).$$

By Lemma [9-](#page-12-0)(c), (d), since *<sup>N</sup>* (R*<sup>d</sup>* ) is Poisson(*N*)-distributed, <sup>P</sup>(*<sup>N</sup>* (R*<sup>d</sup>* ) <sup>≥</sup> *<sup>N</sup>*(*cx*2*<sup>p</sup>*<sup>0</sup> <sup>−</sup> <sup>1</sup>)) <sup>≤</sup> exp(−*N g*(*cx*2*<sup>p</sup>*<sup>0</sup> <sup>−</sup> <sup>1</sup>)) and <sup>P</sup>(|*<sup>N</sup>* (R*<sup>d</sup>* ) <sup>−</sup> *<sup>N</sup>*| ≥ *cNx*) <sup>≤</sup> 2 exp(−*N f* (*cx*)). Next, using that the family (*<sup>N</sup>* (*F*))*F*∈*<sup>P</sup>* is independent, with *<sup>N</sup>* (*F*) Poisson(*N*μ(*F*))-distributed, we use Lemma [9-](#page-12-0)(a) and that #(*P*) <sup>=</sup> <sup>2</sup>*<sup>d</sup>* to obtain, for any θ > 0,

$$\mathbb{E}\left(\exp\left(\theta\sum\_{F\in\mathcal{P}\_{\ell}}|\Pi\_{N}(F)-N\mu(F)|\right)\right)\leq\prod\_{F\in\mathcal{P}\_{\ell}}2^{N\mu(F)(\epsilon^{\theta}-\theta-1)}\leq 2^{2^{\ell\ell}}e^{N(\epsilon^{\theta}-\theta-1)}.$$

Hence

$$\begin{aligned} &\mathbb{P}\left(\sum\_{F\in\mathcal{P}\_{\ell}}|\Pi\_{N}(F)-N\mu(F)|\geq cN\varkappa2^{p\ell}r\_{\ell}\right) \\ &\leq \exp\left(-c\theta N\varkappa2^{p\ell}r\_{\ell}\right)2^{2^{d\ell}}\exp\left(N(e^{\theta}-\theta-1)\right). \end{aligned}$$

Choosing <sup>θ</sup> <sup>=</sup> log(<sup>1</sup> <sup>+</sup> *cx*2*<sup>p</sup><sup>r</sup>*), we find

$$\mathbb{P}\left(\sum\_{F\in\mathcal{P}\_{\ell}}|\Pi\_{N}(F)-N\mu(F)|\geq cN\varkappa2^{p\ell}r\_{\ell}\right)\leq 2^{2^{d\ell}}\exp(-Nf(c\infty2^{p\ell}r\_{\ell})).$$

We have checked that

$$\begin{aligned} \varepsilon(N, \mathbf{x}) &\le 2 \exp(-Nf(c\mathbf{x})) + \exp(-Ng(c\mathbf{x}2^{p\ell\_0} - 1)), \\ &+ \sum\_{\ell=1}^{\ell\_0} 2^{2^{d\ell}} \exp(-Nf(c\mathbf{x}2^{p\ell}r\_\ell)). \end{aligned}$$

At this point, the value of *c* > 0 is not allowed to vary anymore. We introduce some other positive constants *a* whose value may change from line to line.

**Case 1:** *cx* <sup>&</sup>gt; 2. Then we choose <sup>0</sup> <sup>=</sup> 1 and *<sup>r</sup>*<sup>1</sup> <sup>=</sup> 1. We have *cx*2*<sup>p</sup>*<sup>0</sup> <sup>−</sup><sup>1</sup> <sup>=</sup> <sup>2</sup>*pcx* <sup>−</sup> <sup>1</sup> <sup>≥</sup> (2*<sup>p</sup>* <sup>−</sup> <sup>1</sup>)*cx* <sup>+</sup> 1 whence *<sup>g</sup>*(*cx*2*<sup>p</sup>*<sup>0</sup> <sup>−</sup> <sup>1</sup>) <sup>≥</sup> *<sup>g</sup>*((2*<sup>p</sup>* <sup>−</sup> <sup>1</sup>)*cx* <sup>+</sup> <sup>1</sup>) <sup>=</sup> *<sup>f</sup>* ((2*<sup>p</sup>* <sup>−</sup> <sup>1</sup>)*cx*). We also have <sup>0</sup> <sup>=</sup><sup>1</sup> <sup>2</sup>2*<sup>d</sup>* exp(−*N f* (*cx*2*<sup>p</sup><sup>r</sup>*)) <sup>=</sup> <sup>2</sup>2*<sup>d</sup>* exp(−*N f* (2*pcx*)). We finally get ε(*N*, *x*) ≤ *C* exp(−*N f* (*ax*)), which proves the statement (in the three cases, when *cx* > 2).

**Case 2:** *cx* <sup>≤</sup> 2. We choose <sup>0</sup> so that (<sup>1</sup> <sup>+</sup> <sup>2</sup>/(*cx*)) <sup>≤</sup> <sup>2</sup>*<sup>p</sup>*<sup>0</sup> <sup>≤</sup> <sup>2</sup>*p*(<sup>1</sup> <sup>+</sup> <sup>2</sup>/(*cx*)), i.e.

$$\ell\_0 := \lfloor \log(1 + 2/(c\chi))/(p\log 2) \rfloor + 1.$$

This implies that *cx*2*<sup>p</sup>*<sup>0</sup> <sup>≥</sup> <sup>2</sup> <sup>+</sup> *cx*. Hence *<sup>g</sup>*(*cx*2*<sup>p</sup>*<sup>0</sup> <sup>−</sup> <sup>1</sup>) <sup>≥</sup> *<sup>g</sup>*(<sup>1</sup> <sup>+</sup> *cx*) <sup>=</sup> *<sup>f</sup>* (*cx*). Furthermore, we have *cx*2*<sup>p</sup><sup>r</sup>* <sup>≤</sup> *cx*2*<sup>p</sup>*<sup>0</sup> <sup>≤</sup> <sup>2</sup>*p*(2+*cx*) <sup>≤</sup> <sup>2</sup>*p*+<sup>2</sup> for all <sup>≤</sup> 0, whence *<sup>f</sup>* (*cx*2*<sup>p</sup><sup>r</sup>*) <sup>≥</sup> *ax*222*<sup>p</sup><sup>r</sup>* <sup>2</sup> (because *<sup>f</sup>* (*x*) <sup>≥</sup> *ax*<sup>2</sup> for all *<sup>x</sup>* ∈ [0, <sup>2</sup>*p*+2]). We thus end up with (we use that 22*<sup>d</sup>* <sup>≤</sup> exp(2*<sup>d</sup>*))

$$\log(N,\chi) \le 3\exp(-Nf(c\chi)) + \sum\_{\ell=1}^{\ell\_0} \exp\left(2^{d\ell} - Na\alpha^2 2^{2p\ell} r\_\ell^2\right).$$

Now the value of *a* > 0 is not allowed to vary anymore, and we introduce *a* > 0, whose value may change from line to line.

*Case 1.1: p* <sup>&</sup>gt; *<sup>d</sup>*/2. We take *<sup>r</sup>* := (<sup>1</sup> <sup>−</sup> <sup>2</sup>−η)2−<sup>η</sup> for some η > 0 such that <sup>2</sup>(*<sup>p</sup>* <sup>−</sup> η) > *<sup>d</sup>*. If *N x*<sup>2</sup> <sup>≥</sup> 1, we easily get

$$\begin{aligned} \varepsilon(N, \mathbf{x}) &\leq 3 \exp(-Nf(c\mathbf{x})) + \sum\_{\ell=1}^{\ell\_0} \exp(2^{d\ell} - Na'\mathbf{x}^2 2^{2(p-\eta)\ell}) \\ &\leq 3 \exp(-Nf(c\mathbf{x})) + C \exp(-a'N\mathbf{x}^2) \\ &\leq C \exp(-Nf(a'\mathbf{x})) .\end{aligned}$$

The last inequality uses that *<sup>y</sup>*<sup>2</sup> <sup>≥</sup> *<sup>f</sup>* (*y*)for all *<sup>y</sup>* <sup>&</sup>gt; 0. If finally *N x*<sup>2</sup> <sup>≤</sup> 1, we obviously have

$$
\varepsilon(N, x) \le 1 \le \exp(1 - Nx^2) \le C \exp(-Nx^2) \le C \exp(-Nf(x)).
$$

We thus always have ε(*N*, *x*) ≤ *C* exp(−*N f* (*a x*)) as desired.

*Case 2.2: p* <sup>=</sup> *<sup>d</sup>*/2. We choose *<sup>r</sup>* := <sup>1</sup>/0. Thus, if *aN*(*x*/0)<sup>2</sup> <sup>≥</sup> 2, we easily find

$$\begin{aligned} \varepsilon(N, \mathbf{x}) &\le 3 \exp(-Nf(c\mathbf{x})) + \sum\_{\ell=1}^{\ell\_0} \exp\left(2^{d\ell} (1 - aN(\mathbf{x}/\ell\_0)^2)\right) \\ &\le 3 \exp(-Nf(c\mathbf{x})) + C \exp\left(-a'N(\mathbf{x}/\ell\_0)^2\right) \\ &\le 3 \exp(-Nf(c\mathbf{x})) + C \exp\left(-Nf(a'\mathbf{x}/\ell\_0)\right) \\ &\le C \exp(-Nf(a'\mathbf{x}/\ell\_0)) \end{aligned}$$

because <sup>0</sup> <sup>≥</sup> 1 and *<sup>f</sup>* is increasing. If now *aN*(*x*/0)<sup>2</sup> <sup>&</sup>lt; 2, we just write

$$\mathbb{E}\left(\varepsilon(N,\mathbf{x})\leq 1\leq\exp(2-aN(\mathbf{x}/\ell\_{0})^{2})\leq C\exp(-aN(\mathbf{x}/\ell\_{0})^{2})\leq C\exp(-Nf(a\mathbf{x}/\ell\_{0})).\right)$$

We thus always have ε(*N*, *x*) ≤ *C* exp(−*N f* (*a x*/0)). Using that <sup>0</sup> ≤ *C* log(2 + 1/*x*), we immediately conclude that ε(*N*, *x*) ≤ *C* exp(−*N f* (*a x*/ log(2 + 1/*x*))) as desired.

*Case 2.3: p* <sup>∈</sup> (0, *<sup>d</sup>*/2). We choose *<sup>r</sup>* := <sup>κ</sup>2(*d*/2−*p*)(<sup>−</sup><sup>0</sup>) with <sup>κ</sup> <sup>=</sup> <sup>1</sup>/(1−2*p*−*d*/2). For all ≤ 0,

$$\begin{aligned} 2^{dl} - aN\mathbf{x}^2 2^{p\ell} r\_\ell^2 &= -a\kappa^2 N \mathbf{x}^{d/p} 2^{2p\ell} \left[ 2^{(d-2p)(\ell-\ell\_0)} \mathbf{x}^{2-d/p} - 2^{(d-2p)\ell} / (Na\mathbf{x}^{d/p}) \right] \\ &\le -a\kappa^2 N \mathbf{x}^{d/p} 2^{2p\ell} \left[ b2^{(d-2p)\ell} - 2^{(d-2p)\ell} / (Na\kappa^2 \mathbf{x}^{d/p}) \right] \end{aligned}$$

where the constant *<sup>b</sup>* <sup>&</sup>gt; 0 is such that 2−(*d*−2*p*)<sup>0</sup> <sup>≥</sup> *bxd*/*p*−<sup>2</sup> (the existence of *<sup>b</sup>* is easily checked). Hence if *N a*κ2*xd*/*<sup>p</sup>* <sup>≥</sup> <sup>2</sup>/*b*, we find

$$2^{dl} - aN\varkappa^2 \mathcal{D}^{2p\ell} r\_{\ell}^2 \le -ab\varkappa^2 N \varkappa^{d/p} \mathcal{D}^{d\ell} b/2$$

and thus, still using that *N xd*/*<sup>p</sup>* <sup>≥</sup> <sup>2</sup>/(*ab*κ2),

$$\sum\_{\ell=1}^{\ell\_0} \exp(2^{d\ell} - Nc^2 \mathbf{x}^2 2^{2p\ell} r\_\ell^2) \le C \exp(-a' N \mathbf{x}^{d/p}).$$

Consequently, we have ε(*N*, *x*) ≤ 3 exp(−*N f* (*cx*))+*C* exp(−*a N xd*/*p*)if *N a*κ2*xd*/*<sup>p</sup>* <sup>≥</sup> <sup>2</sup>/*b*. As usual, the case where *N a*κ2*xd*/*<sup>p</sup>* <sup>≤</sup> <sup>2</sup>/*<sup>b</sup>* is trivial, since then

$$
\varepsilon(N, \mathbf{x}) \le 1 \le \exp(2/b - N a \kappa^2 \mathbf{x}^{d/p}) \le \mathbf{C} \exp(-a' N \mathbf{x}^{d/p}).
$$

This ends the proof.

#### <span id="page-15-0"></span>**5 Depoissonization in the compact case**

<span id="page-15-2"></span>We next check the following compact version of Theorem [2.](#page-3-1)

**Proposition 10** *Assume that* μ *is supported in* (−1, 1] *<sup>d</sup> . Let p* <sup>&</sup>gt; <sup>0</sup> *and d* <sup>≥</sup> <sup>1</sup> *be fixed. There are some positive constants C and c* (*depending only on p*, *d*) *such that for all N* ≥ 1*, all x* ∈ (0,∞)*,*

$$\mathbb{P}\left[\mathcal{D}\_{p}(\mu\_{N},\mu)\geq x\right] \leq \mathbf{1}\_{\{x\leq 1\}}C \begin{cases} \exp(-cNx^{2}) & \text{if } p > d/2;\\ \exp\left(-cN(\mathbf{x}/\log(2+1/\mathbf{x}))^{2}\right) & \text{if } p = d/2;\\ \exp\left(-cNx^{d/p}\right) & \text{if } p \in (0,d/2). \end{cases}$$

We will need the following easy remark.

<span id="page-15-1"></span>**Lemma 11** *For all N* ≥ 1*, for X Poisson*(*N*)*-distributed, for all k* ∈ {0,..., <sup>√</sup>*N*}*,*

$$\mathbb{P}[X = N + k] \ge \kappa\_0 N^{-1/2} \quad \text{where } \kappa\_0 = e^{-2}/\sqrt{2}.$$

*Proof* By Perrin [\[34](#page-31-9)], we have *N*! ≤ *e* <sup>√</sup>*N*(*N*/*e*)*<sup>N</sup>* . Thus

$$\begin{aligned} \mathbb{P}[X = N + k] &= e^{-N} \frac{N^{N+k}}{(N+k)!} \ge e^{-N-1} \frac{N^{N+k}}{\sqrt{N+k}((N+k)/e)^{N+k}} \\ &\ge \frac{1}{\sqrt{2N}} \left(\frac{N}{N+k}\right)^{N+k} e^{k-1} .\end{aligned}$$

Since log(<sup>1</sup> <sup>+</sup> *<sup>x</sup>*) <sup>≤</sup> *<sup>x</sup>* on (0, <sup>1</sup>), we have ((*<sup>N</sup>* <sup>+</sup> *<sup>k</sup>*)/*N*)*N*+*<sup>k</sup>* <sup>≤</sup> exp(*<sup>k</sup>* <sup>+</sup> *<sup>k</sup>*2/*N*) <sup>≤</sup> exp(*<sup>k</sup>* <sup>+</sup> <sup>1</sup>), so that <sup>P</sup>[*<sup>X</sup>* <sup>=</sup> *<sup>N</sup>* <sup>+</sup> *<sup>k</sup>*] ≥ *<sup>e</sup>*−2/ <sup>√</sup>2*N*.

*Proof of Proposition 10* The probability indeed vanishes if *x* > 1, since *D<sup>p</sup>* is smaller than 1 when restricted to probability measures on (−1, 1] *<sup>d</sup>* . In the sequel, the constants may only depend on *p* and *d*.

**Step 1.** We introduce a Poisson measure *<sup>N</sup>* on R*<sup>d</sup>* with intensity measure *N*μ and the associated empirical measure *<sup>N</sup>* <sup>=</sup> *<sup>N</sup>* /*<sup>N</sup>* (R*<sup>d</sup>* ). Conditionally on {*<sup>N</sup>* (R*<sup>d</sup>* ) <sup>=</sup> *n*}, *<sup>N</sup>* has the same law as μ*<sup>n</sup>* (the empirical measure of *n* i.i.d. random variables with law μ).

$$\mathbb{P}\left[\Pi\_N(\mathbb{R}^d)\mathcal{D}\_p(\Psi\_N,\mu)\geq N\mathbf{x}\right] = \sum\_{n\geq 0} \mathbb{P}\left[\Pi\_N(\mathbb{R}^d) = n\right] \mathbb{P}\left[n\mathcal{D}\_p(\mu\_n,\mu)\geq N\mathbf{x}\right].$$

By Lemma [11](#page-15-1) (since *<sup>N</sup>* (R*<sup>d</sup>* ) is Poisson(*N*)-distributed),

$$\frac{1}{\sqrt{N}}\sum\_{k=0}^{\lfloor\sqrt{N}\rfloor}\mathbb{P}\left[ (N+k)\mathcal{D}\_p(\mu\_{N+k},\mu) \ge Nx \right] \le \kappa\_0^{-1}\mathbb{P}\left[ \Pi\_N(\mathbb{R}^d)\mathcal{D}\_p(\Psi\_N,\mu) \ge Nx \right],$$

which of course implies that (for all *N* ≥ 1, all *x* > 0),

$$\frac{1}{\sqrt{N}}\sum\_{k=0}^{\lfloor\sqrt{N}\rfloor}\mathbb{P}\left[\mathcal{D}\_{p}(\mu\_{N+k},\mu)\geq x\right]\leq\kappa\_{0}^{-1}\mathbb{P}\left[\Pi\_{N}(\mathbb{R}^{d})\mathcal{D}\_{p}(\Psi\_{N},\mu)\geq Nx\right].$$

**Step 2.** Here we prove that there is a constant *A* > 0 such that for any *N* ≥ 1, any *k* ∈ {0,..., <sup>√</sup>*N*}, any *<sup>x</sup>* <sup>&</sup>gt; *AN*−1/2,

$$\mathbb{P}\left[\mathcal{D}\_p(\mu\_N, \mu) \ge \mathbf{x}\right] \le \mathbb{P}\left[\mathcal{D}\_p(\mu\_{N+k}, \mu) \ge \mathbf{x}/2\right].$$

Build μ*<sup>n</sup>* for all values of *n* ≥ 1 with the same i.i.d. family of μ-distributed random variables (*Xk* )*k*≥1. Then a.s.,

$$|\mu\_{N+k} - \mu\_N|\_{TV} \le \left| \frac{k}{N(N+k)} \sum\_{1}^{N} \delta\_{X\_j} \right|\_{TV} + \left| \frac{1}{N+k} \sum\_{N+1}^{N+k} \delta\_{X\_j} \right|\_{TV} \le \frac{k}{N+k} \le \frac{1}{\sqrt{N}}.$$

This obviously implies (recall Notation [4-](#page-6-1)(a)) that *<sup>D</sup>p*(μ*<sup>N</sup>* , μ*N*+*<sup>k</sup>* ) <sup>≤</sup> *C N*−1/<sup>2</sup> a.s. (where *C* depends only on *p*). By the triangular inequality, *Dp*(μ*<sup>N</sup>* , μ) ≤ *<sup>D</sup>p*(μ*N*+*<sup>k</sup>* , μ) <sup>+</sup> *C N*−1/2, whence

$$\mathbb{P}\left[\mathcal{D}\_{p}(\mu\_{N},\mu)\geq\mathbf{x}\right] \leq \mathbb{P}\left[\mathcal{D}\_{p}(\mu\_{N+k},\mu)\geq\mathbf{x} - \mathbf{C}N^{-1/2}\right] \leq \mathbb{P}\left[\mathcal{D}\_{p}(\mu\_{N+k},\mu)\geq\mathbf{x}/2\right]$$

if *<sup>x</sup>* <sup>−</sup> *C N*−1/<sup>2</sup> <sup>≥</sup> *<sup>x</sup>*/2, i.e. *<sup>x</sup>* <sup>≥</sup> <sup>2</sup>*C N*−1/2.

**Step 3.** Gathering Steps 1 and 2, we deduce that for all *<sup>N</sup>* <sup>≥</sup> 1, all *<sup>x</sup>* <sup>&</sup>gt; *AN*−1/2,

$$\begin{aligned} \mathbb{P}\left[\mathcal{D}\_p(\mu\_N, \mu) \ge x\right] &\le \frac{1}{\sqrt{N}} \sum\_{k=0}^{\lfloor \sqrt{N} \rfloor} \mathbb{P}\left[\mathcal{D}\_p(\mu\_{N+k}, \mu) \ge x/2\right] \\ &\le C \mathbb{P}\left[\Pi\_N(\mathbb{R}^d) \mathcal{D}\_p(\Psi\_N, \mu) \ge Nx/2\right]. \end{aligned}$$

We next apply Proposition [8.](#page-11-1) Observing that, for *x* ∈ (0, 1],

- (i) exp(−*N f* (*cx*/2)) <sup>≤</sup> exp(−*cNx*2) (case *<sup>p</sup>* <sup>&</sup>gt; *<sup>d</sup>*/2),
- (ii) exp(−*N f* (*cx*/2 log(<sup>2</sup> <sup>+</sup> <sup>2</sup>/*x*))) <sup>≤</sup> exp(−*cN*(*x*/ log(<sup>2</sup> <sup>+</sup> <sup>1</sup>/*x*)2) (case *<sup>p</sup>* <sup>=</sup> *<sup>d</sup>*/2),
- (iii) exp(−*N f* (*cx*/2)) <sup>+</sup> exp(*cN*(*x*/2)*d*/*p*) <sup>≤</sup> exp(−*cNxd*/*p*) (case *<sup>p</sup>* <sup>∈</sup> (0, *<sup>d</sup>*/2))

concludes the proof when *x* > *AN*−1/2. But the other case is trivial, because for *<sup>x</sup>* <sup>≤</sup> *AN*−1/2,

$$\mathbb{P}[\mathcal{D}\_p(\mu\_N, \mu) \ge x] \le 1 \le \exp(A^2 - Nx^2) \le C \exp(-Nx^2),$$

which is also smaller than *<sup>C</sup>* exp(−*N*(*x*/ log(<sup>2</sup> <sup>+</sup> <sup>1</sup>/*x*))2) and than *<sup>C</sup>* exp(−*N xd*/*p*) (if *d* > 2*p*).

#### <span id="page-17-0"></span>**6 Concentration inequalities in the non compact case**

<span id="page-17-1"></span>Here we conclude the proof of Theorem [2.](#page-3-1) We will need some concentration estimates for the Binomial distribution.

**Lemma 12** *Let X be Binomial*(*N*, *p*)*-distributed. Recall that f was defined in Notation* [7](#page-11-2)*.*

$$\begin{array}{l} \text{(a)} \; \mathbb{P}[|X - Np| \ge Npz] \le (\mathbf{1}\_{\{p(1+\varepsilon) \le 1\}} + \mathbf{1}\_{\{\varepsilon \le 1\}}) \exp(-Npf(z)) \text{ for all } z > 0. \\\text{(b)} \; \mathbb{P}[|X - Np| \ge Npz] \le Np \; \text{for all } z > 1. \\\text{(c)} \; \mathbb{E}(\exp(-\theta X)) = (1 - p + pe^{-\theta})^N \le \exp(-Np(1 - e^{-\theta})) \; \text{for } \theta > 0. \end{array}$$

*Proof* Point (c) is straightforward. Point (b) follows from the fact that for *z* > 1, <sup>P</sup>[|*<sup>X</sup>* <sup>−</sup> *N p*| ≥ *Npz*] = <sup>P</sup>[*<sup>X</sup>* <sup>≥</sup> *N p*(<sup>1</sup> <sup>+</sup> *<sup>z</sup>*)] ≤ <sup>P</sup>[*<sup>X</sup>* = <sup>0</sup>] = <sup>1</sup> <sup>−</sup> (<sup>1</sup> <sup>−</sup> *<sup>p</sup>*)*<sup>N</sup>* <sup>≤</sup> *pN*. For point (a), we use Bennett's inequality [\[4](#page-30-18)], see Devroye and Lugosi [\[17,](#page-30-19) Exercise 2.2 page 11], together with the obvious facts that <sup>P</sup>[*<sup>X</sup>* <sup>−</sup> *N p* <sup>≥</sup> *Npz*] = <sup>0</sup> if *<sup>p</sup>*(<sup>1</sup> <sup>+</sup> *<sup>z</sup>*) > 1 and <sup>P</sup>[*<sup>X</sup>* <sup>−</sup> *N p* ≤ −*Npz*] = 0 if *<sup>z</sup>* <sup>&</sup>gt; 1. The following elementary tedious computations also works: write <sup>P</sup>[|*<sup>X</sup>* <sup>−</sup> *N p*| ≥ *Npz*] = <sup>P</sup>(*<sup>X</sup>* <sup>≥</sup> *N p*(<sup>1</sup> <sup>+</sup> *<sup>z</sup>*)) <sup>+</sup> <sup>P</sup>(*<sup>N</sup>* <sup>−</sup> *<sup>X</sup>* <sup>≥</sup> *<sup>N</sup>*(<sup>1</sup> <sup>−</sup> *<sup>p</sup>* <sup>+</sup> *zp*)) =: (*p*,*z*) <sup>+</sup> (<sup>1</sup> <sup>−</sup> *<sup>p</sup>*,*zp*/(<sup>1</sup> <sup>−</sup> *<sup>p</sup>*)), observe that *N* − *X* ∼ Binomial(*N*, 1 − *p*). Use that (*p*,*z*) ≤ **1**{*p*(1+*z*)≤1} exp(−θ *N p*(1 + *<sup>z</sup>*))(<sup>1</sup> <sup>−</sup> *<sup>p</sup>* <sup>+</sup> *pe*<sup>θ</sup> )*<sup>N</sup>* and choose <sup>θ</sup> <sup>=</sup> log((<sup>1</sup> <sup>−</sup> *<sup>p</sup>*)(<sup>1</sup> <sup>+</sup> *<sup>z</sup>*)/(<sup>1</sup> <sup>−</sup> *<sup>p</sup>* <sup>−</sup> *pz*)), this gives (*p*,*z*) ≤ **1**{*p*(1+*z*)≤1} exp(−*N*[*p*(1 + *z*)log(1 + *z*) + (1 − *p* − *pz*)log((1 − *p* − *pz*)/(1− *p*))]). A tedious study shows that (*p*,*z*) ≤ **1**{*p*(1+*z*)≤1} exp(−*Npf* (*z*)) and that (1 − *p*,*zp*/(1 − *p*)) ≤ **1**{*z*≤1} exp(−*Npf* (*z*)).

We next estimate the first term when computing *Dp*(μ*<sup>N</sup>* , μ).

<span id="page-18-0"></span>**Lemma 13** *Let* <sup>μ</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ) *and p* <sup>&</sup>gt; <sup>0</sup>*. Assume* [\(1\)](#page-3-3)*,* [\(2\)](#page-3-0) *or* [\(3\)](#page-3-2)*. Recall Notation* [4](#page-6-1) *and put Z <sup>p</sup> <sup>N</sup>* := *<sup>n</sup>*≥<sup>0</sup> <sup>2</sup>*pn*|μ*<sup>N</sup>* (*Bn*) <sup>−</sup> μ(*Bn*)|*. Let x*<sup>0</sup> *be fixed. For all x* <sup>&</sup>gt; <sup>0</sup>*,*

$$\begin{aligned} \mathbb{P}[Z\_N^P \ge x] &\le C \exp(-cNx^2) \mathbf{1}\_{\{x \le x\_0\}} \\ &+ C \begin{cases} \exp(-cNx^{a/p}) \mathbf{1}\_{\{x \ge x\_0\}} & \text{under}(1), \\ \exp(-c(Nx)^{(a-\varepsilon)/p}) \mathbf{1}\_{\{x \le x\_0\}} + \exp(-c(Nx)^{a/p}) \mathbf{1}\_{\{x > x\_0\}} & \forall \,\varepsilon \in (0,\alpha) \,\text{under}(2), \\ N(Nx)^{-(q-\varepsilon)/p} & \forall \,\varepsilon \in (0,q) \,\text{under}(3). \end{cases} \end{aligned}$$

*The positive constants C and c depend only on p*, *d*, *x*<sup>0</sup> *and either on* α, γ , *E*α,γ (μ) (*under* (2)) *or on* α, γ , *E*α,γ (μ), ε (*under* (2)) *or on q*, *Mq* (μ), ε *(under* (3))*.*

*Proof* During the proof, the constants are only allowed to depend on the same quantities as in the statement, unless we precise it. Under (1) or (2), we assume that γ = 1 without loss of generality (by scaling), whence *E*α,1(μ) < ∞ and thus μ(*Bn*) <sup>≤</sup> *Ce*−2(*n*−1)α for all *<sup>n</sup>* <sup>≥</sup> 0. Under (3), we have μ(*Bn*) <sup>≤</sup> *<sup>C</sup>*2−*qn* for all *n* ≥ 0. For η > 0 to be chosen later (observe that *<sup>n</sup>*≥0(<sup>1</sup> <sup>−</sup> <sup>2</sup>−η)2−η*<sup>n</sup>* <sup>=</sup> 1), putting *<sup>c</sup>* := <sup>1</sup> <sup>−</sup> <sup>2</sup>−<sup>η</sup> and *zn* := *cx*2−(*p*+η)*n*/μ(*Bn*),

$$\begin{split} \mathbb{P}\left(Z\_{N}^{p} \geq x\right) &\leq \left(\sum\_{n\geq 0} \mathbf{1}\_{\{z\_{n}\leq 2\}} \mathbb{P}\left[|N\mu\_{N}(B\_{n}) - N\mu(B\_{n})| \geq N\mu(B\_{n})z\_{n}\right]\right) \wedge 1 \\ &\quad + \left(\sum\_{n\geq 0} \mathbf{1}\_{\{z\_{n}>2\}} \mathbb{P}\left[|N\mu\_{N}(B\_{n}) - N\mu(B\_{n})| \geq N\mu(B\_{n})z\_{n}\right]\right) \wedge 1 \\ &=: \left(\sum\_{n\geq 0} I\_{n}(N,x)\right) \wedge 1 + \left(\sum\_{n\geq 0} J\_{n}(N,x)\right) \wedge 1. \end{split}$$

From now on, the value of *c* > 0 is not allowed to vary anymore. We introduce another positive constant *a* > 0 whose value may change from line to line.

**Step 1: bound of** *In*. Here we show that under (3) (which is of course implied by (1) or (2)), if η ∈ (0, *q*/2 − *p*), there is *A*<sup>0</sup> > 0 such that

$$\sum\_{n\geq 0} I\_n(N, \boldsymbol{x}) \leq C \exp(-aN\boldsymbol{x}^2) \mathbf{1}\_{\{\boldsymbol{x} \leq A\_0\}} \quad \text{if } N\boldsymbol{x}^2 \geq 1.$$

This will obviously imply that for all *N* ≥ 1, all *x* > 0,

$$\left(\sum\_{n\geq 0} I\_n(N, \ge)\right) \land 1 \leq C \exp(-aN\pi^2) \mathbf{1}\_{\{\boldsymbol{\chi} \leq A\_0\}}.$$

First, *<sup>n</sup>*≥<sup>0</sup> *In*(*N*, *<sup>x</sup>*) <sup>=</sup> 0 if *zn* <sup>&</sup>gt; 2 for all *<sup>n</sup>* <sup>≥</sup> 0. Recalling that μ(*Bn*) <sup>≤</sup> *<sup>C</sup>*2−*qn*, this is the case if *<sup>x</sup>* <sup>≥</sup> (2*C*/*c*)sup*n*≥<sup>0</sup> <sup>2</sup>(*p*+η−*q*)*<sup>n</sup>* <sup>=</sup> (2*C*/*c*) := *<sup>A</sup>*0. Next, since *N*μ*<sup>N</sup>* (*Bn*) ∼ Binomial(*N*, μ(*Bn*)), Lemma [12-](#page-17-1)(a) leads us to

$$I\_n(N, \ge) \le 2\mathbf{1}\_{\{z\_n \le 2\}} \exp(-N\mu(B\_n)f(z\_n)) \le 2\exp(-N\mu(B\_n)z\_n^2/4)),$$

because *<sup>f</sup>* (*x*) <sup>≥</sup> *<sup>x</sup>*2/4 for *<sup>x</sup>* ∈ [0, <sup>2</sup>]. Since finally μ(*Bn*)*z*<sup>2</sup> *<sup>n</sup>*/<sup>4</sup> <sup>≥</sup> *ax*22(*q*−2*p*−2η)*n*, we easily conclude, since *<sup>q</sup>* <sup>−</sup> <sup>2</sup>*<sup>p</sup>* <sup>−</sup> <sup>2</sup>η > 0 and since *N x*<sup>2</sup> <sup>≥</sup> 1, that

$$\sum\_{n\geq 0} I\_n(N, \mathbf{x}) \leq C \sum\_{n\geq 0} \exp(-aN\mathbf{x}^2 2^{(q-2p-2\eta)n}) \mathbf{1}\_{\{\mathbf{x} \leq A\_0\}} \leq C \exp(-aN\mathbf{x}^2) \mathbf{1}\_{\{\mathbf{x} \leq A\_0\}}.$$

**Step 2: bound of** *Jn***under** (1) **or** (2) **when** *x* ≤ *A*. Here we fix *A* > 0 and prove that if η > 0 is small enough, for all *<sup>x</sup>* <sup>∈</sup> (0, *<sup>A</sup>*] such that *N x*<sup>2</sup> <sup>≥</sup> 1,

- *n*≥0 *Jn*(*N*, *x*) ≤*C* exp(−*aNx*2) under (1), exp(−*aNx*2) <sup>+</sup> exp(−*a*(*N x*)(α−ε)/*p*) <sup>∀</sup> <sup>ε</sup> <sup>∈</sup> (0, α) under (2).

Here the positive constants *C* and *a* are allowed to depend additionally on *A*. This will imply, as usual, that for all *N* ≥ 1, all *x* ∈ (0, *A*],

$$\left(\sum\_{n\geq 0} J\_{\mathbb{H}}(N, \mathbf{x})\right) \wedge 1 \leq \mathcal{C} \begin{cases} \exp(-aN\mathbf{x}^{2}) & \text{under (1),} \\ \exp(-aN\mathbf{x}^{2}) + \exp(-a(N\mathbf{x})^{(\alpha - \varepsilon)/p}) & \forall \,\varepsilon \in (0, \alpha) \text{ under (2)}. \end{cases}$$

By Lemma [12-](#page-17-1)(a), (b) (since *zn* > 2 implies **1**{μ(*Bn* )(1+*zn* )≤1} + **1**{*zn*≤1} ≤ **1**{*zn*≤1/μ(*Bn* )}),

$$\begin{aligned} J\_n(N, \boldsymbol{x}) &\leq \mathbf{1}\_{\{2 < \boldsymbol{z}\_n \leq 1/\mu(B\_n)\}} \min\left\{ \exp(-N\mu(B\_n)f(\boldsymbol{z}\_n)), N\mu(B\_n) \right\} \\ &\leq \mathbf{1}\_{\{\boldsymbol{z}\_n\mu(B\_n) \leq 1\}} \min\left\{ \exp\left(-nN\mu(B\_n)\boldsymbol{z}\_n \log[2\vee z\_n]\right), N\mu(B\_n) \right\} \end{aligned}$$

because *<sup>f</sup>* (*y*) <sup>≥</sup> *ay* log *<sup>y</sup>* <sup>≥</sup> *ay* log[<sup>2</sup> <sup>∨</sup> *<sup>y</sup>*] for *<sup>y</sup>* <sup>&</sup>gt; 2. Since μ(*Bn*) <sup>≤</sup> *Ce*−2(*n*−1)α , we get

$$J\_n(N, x) \le C \min \{ \exp(-aNx2^{-(p+\eta)\mu} \log[2 \vee (a\ge 2^{-(p+\eta)\mu} e^{2^{(n-1)\mu}})]), \, N e^{2^{-(n-1)\mu}} \}.$$

A straightforward computation shows that there is a constant *K* such that for *n* ≥ *n*<sup>1</sup> := *K*(1+log log(*K*/*x*)), we have log(*ax*2−(*p*+η)*ne*2(*n*−1)α ) <sup>≥</sup> <sup>2</sup>(*n*−<sup>1</sup>)α/2. Consequently,

$$\begin{aligned} \sum\_{n\geq 0} J\_n(N, x) &\leq C n\_1 \exp(-a N x 2^{-(p+\eta)n\_1}) \\ &+ C \sum\_{n>n\_1} \min\left\{ \exp(-a N x 2^{(\alpha - p - \eta)n}), e^{-2^{(\alpha - 1)\alpha}} \right\} \\ &= C J^1(N, x) + C J^2(N, x). \end{aligned}$$

We first show that *<sup>J</sup>* <sup>1</sup>(*N*, *<sup>x</sup>*) <sup>≤</sup> *Ce*−*aNx*<sup>2</sup> (here we actually could get something much better). First, since *n*<sup>1</sup> = *K* + *K* log log(*K*/*x*) and *x* ∈ [0, *A*], we clearly have e.g. *<sup>x</sup>*2−(*p*+η)*n*<sup>1</sup> <sup>≥</sup> *ax*3/2. Next, *N x*<sup>2</sup> <sup>≥</sup> 1 implies that 1/*<sup>x</sup>* <sup>≤</sup> (*N x*3/2)2. Thus

$$\begin{aligned} J^1(N, \boldsymbol{x}) &\leq C(1 + \log \log(C(N\boldsymbol{x}^{3/2})^2)) \exp(-aN\boldsymbol{x}^{3/2}) \\ &\leq C \exp(-aN\boldsymbol{x}^{3/2}) \leq \exp(-aN\boldsymbol{x}^2). \end{aligned}$$

We now treat *J* <sup>2</sup>(*N*, *x*).

*Step 2.1.* Under (1), we immediately get, if η ∈ (0, α − *p*) (recall that *x* ∈ [0, *A*]),

$$J^2(N, \chi) \le \sum\_{n\ge 0} \exp\left(-aN\chi 2^{(\alpha - p - \eta)n}\right) \le C\exp(-aN\chi) \le C\exp(-aN\chi^2),$$

where we used that *<sup>x</sup>* <sup>≤</sup> *<sup>A</sup>* and *N x*<sup>2</sup> <sup>≥</sup> 1 (whence *N x* <sup>≥</sup> <sup>1</sup>/*A*).

*Step 2.2.* Under (2), we first write

$$J^2(N, \chi) \le \sum\_{n\ge 0} \min\left\{ \exp(-aN\chi 2^{(\alpha - p - \eta)n}), e^{-2^{(\alpha - 1)a}} \right\}.$$

$$\le n\_2 \exp(-cN\chi 2^{(\alpha - p - \eta)n\_2}) + N e^{-2^{(n\_2 - 1)a}}.$$

We choose *<sup>n</sup>*<sup>2</sup> := log(*N x*)/((*<sup>p</sup>* <sup>+</sup> η)log 2), which yields us to 2(*n*2−1)α <sup>≥</sup> (*N x*)α/(*<sup>p</sup>*+η)/22<sup>α</sup> and (*N x*)2(α−*p*−η)*n*<sup>2</sup> <sup>≤</sup> (*N x*)α/(*p*+η). Consequently (recall that *x* ∈ (0, *A*]),

$$\begin{aligned} J^2(N, \mathbf{x}) &\leq C(\mathbf{l} + \log(N\mathbf{x}) + N) \exp(-a(N\mathbf{x})^{\alpha/(p+\eta)}), \\ &\leq C(\mathbf{l} + N) \exp(-a(N\mathbf{x})^{\alpha/(p+\eta)}). \end{aligned}$$

For any fixed ε ∈ (0, α), we choose η > 0 small enough so that α/(*p*+η) ≥ (α−ε)/*p* and we conclude that (recall that *N x* <sup>≥</sup> <sup>1</sup>/*<sup>A</sup>* because *N x*<sup>2</sup> <sup>≥</sup> 1 and *<sup>x</sup>* <sup>≤</sup> *<sup>A</sup>*)

$$J^2(N, \mathbf{x}) \le \mathcal{C}(1+N) \exp(-a(N\mathbf{x})^{(\alpha-\varepsilon)/p}) \le \mathcal{C} \exp(-a(N\mathbf{x})^{(\alpha-\varepsilon)/p}).$$

The last inequality is easily checked, using that *N x*<sup>2</sup> <sup>≥</sup> 1 implies that *<sup>N</sup>* <sup>≤</sup> (*N x*)2.

**Step 3: bound of** *Jn***under** (3). Here we show that for all ε ∈ (0, *q*), if η > 0 is small enough,

$$\sum\_{n\geq 0} J\_n(N, \chi) \leq CN \left(\frac{1}{N\chi}\right)^{(q-\varepsilon)/p} \quad \text{if } N\chi \geq 1.$$

As usual, this will imply that for all *x* > 0, all *N* ≥ 1,

$$\left(\sum\_{n\geq 0} J\_n(N,\alpha)\right) \wedge 1 \leq CN\left(\frac{1}{N\alpha}\right)^{(q-\varepsilon)/p}.$$

Exactly as in Step 2, we get from Lemma [12-](#page-17-1)(a)–(b) that

$$J\_n(N, \ge) \le \min\left\{ \exp\left(-aN\mu(B\_n)z\_n \log[2\vee z\_n]\right), N\mu(B\_n) \right\}.$$

Hence for *<sup>n</sup>*<sup>3</sup> to be chosen later, since *aN*μ(*Bn*)*zn* <sup>=</sup> *aNx*2−(*p*+η)*n*,

$$\begin{aligned} \sum\_{n\geq 0} J\_n(N, \chi) &\leq C \sum\_{n=0}^{n\_3} \exp(-aN\chi 2^{-(p+\eta)n}) + CN \sum\_{n>n\_3} 2^{-qn} \\ &\leq Cn \exp(-aN\chi 2^{-(p+\eta)n\_3}) + CN 2^{-qn\_3} .\end{aligned}$$

We choose *<sup>n</sup>*<sup>3</sup> := (*<sup>q</sup>* <sup>−</sup> ε)log(*N x*)/(*pq* log 2), which implies that 2−*qn*<sup>3</sup> <sup>≤</sup> <sup>2</sup>*<sup>q</sup>* (*N x*)−(*q*−ε)/*<sup>p</sup>* and that 2−(*p*+η)*n*<sup>3</sup> <sup>≥</sup> (*N x*)−(*q*−ε)(*p*+η)/(*pq*) . Hence

$$\sum\_{n\geq 0} J\_n(N, \mathbf{x}) \leq C \log(N\mathbf{x}) \exp(-a(N\mathbf{x})^{1-(q-\varepsilon)(p+\eta)/(pq)}) + CN(N\mathbf{x})^{-(q-\varepsilon)/p}.$$

If η ∈ (0, *p*ε/(*q* − ε)), then 1 − (*q* − ε)(*p* + η)/(*pq*) > 0, and thus

$$(\log(N\boldsymbol{\chi})\exp(-a(N\boldsymbol{\chi})^{1-(q-\varepsilon)(p+\eta)/(pq)}) \leq C(N\boldsymbol{\chi})^{-(q-\varepsilon)/p}.$$

This ends the step.

**Step 4.** We next assume [\(1\)](#page-3-3) and prove that for all *<sup>x</sup>* <sup>≥</sup> *<sup>A</sup>*<sup>1</sup> := <sup>2</sup>*p*[*Mp*(μ) <sup>+</sup> (2 log *<sup>E</sup>*α,1(μ))*p*/α],

$$\Pr[Z\_N^p \ge x] \le C \exp(-aNx^{\alpha/p}).$$

A simple computation shows that for any <sup>ν</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ), *<sup>n</sup>*≥<sup>0</sup> <sup>2</sup>*pn*ν(*Bn*) <sup>≤</sup> <sup>2</sup>*<sup>p</sup> Mp*(ν), whence *Z <sup>p</sup> <sup>N</sup>* <sup>≤</sup> <sup>2</sup>*<sup>p</sup> Mp*(μ)+2*pN*−<sup>1</sup> *<sup>N</sup>* <sup>1</sup> <sup>|</sup>*Xi*|*<sup>p</sup>* <sup>≤</sup> <sup>2</sup>*<sup>p</sup> Mp*(μ)+2*p*[*N*−<sup>1</sup> *<sup>N</sup>* <sup>1</sup> |*Xi*| <sup>α</sup>]*p*/α. Thus

$$\Pr[Z\_N^p \ge x] \le \Pr\left[N^{-1} \sum\_{l}^{N} |X\_l|^{\alpha} \ge \lceil x \mathcal{Z}^{-p} - M\_p(\mu) \rceil^{\alpha/p}\right].$$

Next, we note that for *y* ≥ 2 log *E*α,1(μ),

$$\Pr\left[N^{-1}\sum\_{1}^{N}|X\_{l}|^{\alpha}\geq\mathbf{y}\right]\leq\exp(-N\mathbf{y}+N\log\mathcal{E}\_{\alpha,1}(\mu))\leq\exp(-N\mathbf{y}/2).$$

The conclusion easily follows, since *<sup>x</sup>* <sup>≥</sup> *<sup>A</sup>*<sup>1</sup> implies that *<sup>y</sup>* := [*x*2−*<sup>p</sup>* <sup>−</sup> *Mp*(μ)] α/*<sup>p</sup>* <sup>≥</sup> 2 log *<sup>E</sup>*α,1(μ) and since *<sup>y</sup>* ≥ [*x*2−*p*−1] α/*<sup>p</sup>* − [*Mp*(μ)] α/*p*.

**Step 5.** Assume [\(2\)](#page-3-0) and put δ := 2*p*/α − 1. Here we show that for all *x* > 0, *N* ≥ 1,

$$\Pr[Z\_N^p \ge x] \le C \exp(-a(Nx)^{a/p}) + C \exp(-aNx^2(\log(1+N))^{-\delta}).$$

*Step 5.1.* For *R* > 0 (large) to be chosen later, we introduce the probability measure <sup>μ</sup>*<sup>R</sup>* as the law of *<sup>X</sup>***1**{|*X*|≤*R*}. We also denote by <sup>μ</sup>*<sup>R</sup> <sup>N</sup>* the corresponding empirical measure (coupled with μ*<sup>N</sup>* in that the *Xi*'s are used for μ*<sup>N</sup>* and the *Xi***1**{|*Xi* |≤*R*}'s are chosen for μ*<sup>R</sup> <sup>N</sup>* ). We set *<sup>Z</sup> <sup>p</sup>*,*<sup>R</sup> <sup>N</sup>* := *<sup>n</sup>*≥<sup>0</sup> <sup>2</sup>*pn* μ*R <sup>N</sup>* (*Bn*) <sup>−</sup> <sup>μ</sup>*R*(*Bn*) and first observe that *Z p <sup>N</sup>* <sup>−</sup> *<sup>Z</sup> <sup>p</sup>*,*<sup>R</sup> N* <sup>≤</sup> <sup>2</sup>*pN*−<sup>1</sup> *<sup>N</sup>* <sup>1</sup> <sup>|</sup>*Xi*|*p***1**{|*Xi* <sup>|</sup>>*R*} <sup>+</sup> <sup>2</sup>*<sup>p</sup>* {|*x*|>*R*} <sup>|</sup>*x*|*p*μ(*dx*). On the one hand, {|*x*|>*R*} <sup>|</sup>*x*|*p*μ(*dx*) <sup>≤</sup> exp(−*R*α/2) <sup>|</sup>*x*|*pe*|*x*<sup>|</sup> α/2μ(*dx*) <sup>≤</sup> *<sup>C</sup>* exp(−*R*α/2) by [\(2\)](#page-3-0) (with γ = 1). On the other hand, since α ∈ (0, *p*], *<sup>N</sup>* <sup>1</sup> <sup>|</sup>*Xi*|*p***1**{|*Xi* <sup>|</sup>>*R*} <sup>≤</sup> *<sup>N</sup>* <sup>1</sup> |*Xi*| <sup>α</sup>**1**{|*Xi* <sup>|</sup>>*R*} *p*/α . Hence if *<sup>x</sup>* <sup>≥</sup> *<sup>A</sup>* exp(−*R*α/2), where *<sup>A</sup>* := <sup>2</sup>*p*+1*C*,

$$\begin{split} \Pr\left( \left| Z\_N^p - Z\_N^{p,R} \right| \ge x \right) &\le \Pr\left( N^{-1} \sum\_{l}^N |X\_l|^p \mathbf{1}\_{\{|X\_l| > R\}} \ge x 2^{-p-1} \right) \\ &\le \Pr\left( \sum\_{l}^N |X\_l|^a \mathbf{1}\_{\{|X\_l| > R\}} \ge (Nx2^{-p-1})^{a/p} \right) \\ &\le \exp(-(Nx2^{-p-1})^{a/p}/2) \mathbb{E} \left[ \exp\left( |X\_1|^a \mathbf{1}\_{\{|X\_1| > R\}}/2 \right) \right]^N. \end{split}$$

Observing that <sup>E</sup>[exp(|*X*1<sup>|</sup> <sup>α</sup>**1**{|*X*1|>*R*}/2)] ≤ <sup>1</sup> <sup>+</sup> <sup>E</sup>[exp(|*X*1<sup>|</sup> α/2)**1**{|*X*1|>*R*}] ≤ <sup>1</sup> <sup>+</sup> *<sup>C</sup>* exp(−*R*α/2) by [\(2\)](#page-3-0) and using that log(<sup>1</sup> <sup>+</sup> *<sup>u</sup>*) <sup>≤</sup> *<sup>u</sup>*, we deduce that for all *<sup>x</sup>* <sup>≥</sup> <sup>2</sup>*p*+1*<sup>C</sup>* exp(−*R*α/2),

$$\Pr\left(\left|Z\_N^p - Z\_N^{p,R}\right| \ge x\right) \le \exp\left(-(Nx2^{-p-1})^{a/p}/2 + CN\exp(-R^\alpha/2)\right).$$

<span id="page-22-0"></span>With the choice

$$R := \left(2\log(1+N)\right)^{1/\alpha},\tag{5}$$

we finally find

$$\Pr\left(\left|Z\_N^p - Z\_N^{p,R}\right| \ge x\right) \le \exp\left(-(Nx2^{-p-1})^{a/p}/2 + C\right) \le C\exp\left(-a(Nx)^{a/p}\right),$$

provided *<sup>x</sup>* <sup>≥</sup> *<sup>A</sup>* exp(−*R*α/2), i.e. (*<sup>N</sup>* <sup>+</sup> <sup>1</sup>)*<sup>x</sup>* <sup>≥</sup> *<sup>A</sup>*. As usual, this immediately extends to any value of *x* > 0.

*Step 5.2.* To study *Z <sup>p</sup>*,*<sup>R</sup> <sup>N</sup>* , we first observe that since <sup>μ</sup>*R*(*Bn*) <sup>=</sup> 0 if 2*n*−<sup>1</sup> <sup>≥</sup> *<sup>R</sup>*, we have 2*pn*μ*R*(*Bn*) <sup>≤</sup> (2*R*)*p*−α/22α*n*/2μ*R*(*Bn*) for all *<sup>n</sup>* <sup>≥</sup> 0. Hence *<sup>Z</sup> <sup>p</sup>*,*<sup>R</sup> N* ≤ (2*R*)*p*−α/2*Z*α/2,*<sup>R</sup> <sup>N</sup>* . But <sup>μ</sup>*<sup>R</sup>* satisfies <sup>R</sup>*<sup>d</sup>* exp(|*x*| α)μ*<sup>R</sup>*(*dx*) < <sup>∞</sup> uniformly in *<sup>R</sup>*, so that we may use Steps 1, 2 and 4 (with *p* = α/2 < α) to deduce that for all *x* > 0, Pr *Z*α/2,*<sup>R</sup> <sup>N</sup>* ≥ *x* <sup>≤</sup> *<sup>C</sup>* exp(−*aNx*2). Consequently, Pr *Z <sup>p</sup>*,*<sup>R</sup> <sup>N</sup>* ≥ *x* ≤ *<sup>C</sup>* exp(−*aN*(*x*/*Rp*−α/2)2). Recalling [\(5\)](#page-22-0) and that <sup>δ</sup> := <sup>2</sup>*p*/α <sup>−</sup> 1, we see that that Pr *Z <sup>p</sup>*,*<sup>R</sup> <sup>N</sup>* ≥ *x* ≤ *C* exp <sup>−</sup>*aNx*2(log(<sup>1</sup> <sup>+</sup> *<sup>N</sup>*))−<sup>δ</sup> . This ends the step.

**Conclusion.** Recall that *x*<sup>0</sup> > 0 is fixed.

First assume [\(1\)](#page-3-3). By Step 4, Pr *Z p <sup>N</sup>* ≥ *x* <sup>≤</sup> *<sup>C</sup>* exp(−*aNx*α/*p*) for all *<sup>x</sup>* <sup>≥</sup> *<sup>A</sup>*1. We deduce from Steps 1 and 2 that for *x* ∈ (0, *A*1), Pr *Z p <sup>N</sup>* ≥ *x* <sup>≤</sup> *<sup>C</sup>* exp(−*aNx*2). We easily conclude that for all *x* > 0, Pr *Z p <sup>N</sup>* ≥ *x* <sup>≤</sup> *<sup>C</sup>* exp(−*aNx*2)**1**{*x*≤*x*0} <sup>+</sup> *<sup>C</sup>* exp(−*aNx*α/*p*)**1**{*x*>*x*0} as desired.

Assume next [\(2\)](#page-3-0). By Step 5, Pr *Z p <sup>N</sup>* ≥ *x* <sup>≤</sup> *<sup>C</sup>* exp(−*aNx*2(log(<sup>1</sup> <sup>+</sup> *<sup>N</sup>*))−δ) <sup>+</sup> *<sup>C</sup>* exp(−*a*(*N x*)α/*<sup>p</sup>*). But if *<sup>x</sup>* <sup>≥</sup> *<sup>x</sup>*0, we clearly have (*N x*)α/*<sup>p</sup>* <sup>≤</sup> *CNx*2(log(1<sup>+</sup> *<sup>N</sup>*))−<sup>δ</sup> because α < *p*, so that Pr *Z p <sup>N</sup>* ≥ *x* <sup>≤</sup> *<sup>C</sup>* exp(−*a*(*N x*)α/*<sup>p</sup>*). If now *<sup>x</sup>* <sup>≤</sup> *<sup>x</sup>*0, we use Steps 1 and 2 to write Pr *Z p <sup>N</sup>* ≥ *x* <sup>≤</sup> *<sup>C</sup>* exp(−*aNx*2) <sup>+</sup> *<sup>C</sup>* exp(−*a*(*N x*)(α−ε)/*p*).

Assume finally [\(3\)](#page-3-2). By Steps 1 and 3, Pr[*<sup>Z</sup> <sup>p</sup> <sup>N</sup>* <sup>≥</sup> *<sup>x</sup>*] ≤ *<sup>C</sup>* exp(−*aNx*2) <sup>+</sup> *C N*(*N x*)−(*q*−ε)/*<sup>q</sup>* for all *<sup>x</sup>* <sup>&</sup>gt; 0. But if *<sup>x</sup>* <sup>≥</sup> *<sup>x</sup>*0, exp(−*aNx*2) <sup>≤</sup> exp(−*aNx*) <sup>≤</sup> *<sup>C</sup>*(*N x*)−(*q*−ε)/*<sup>q</sup>* <sup>≤</sup> *C N*(*N x*)−(*q*−ε)/*<sup>q</sup>* . We conclude that for all *<sup>x</sup>* <sup>&</sup>gt; 0, Pr[*<sup>Z</sup> <sup>p</sup> <sup>N</sup>* ≥ *x*] ≤ *<sup>C</sup>* exp(−*aNx*2)**1**{*x*≤*x*0} <sup>+</sup> *C N*(*N x*)−(*q*−ε)/*<sup>q</sup>* as desired.

We can now give the

*Proof of Theorem 2* Let us recall that the constants during this proof may depend only on *p*, *d* and either on α, γ , *E*α,γ (μ) (under (1)) or on α, γ , *E*α,γ (μ), ε (under (2)) or on *q*, *Mq* (μ), ε (under (3)).

Using Lemma [5,](#page-6-0) we write

$$\begin{split} T\_p(\mu\_N, \mu) &\leq \kappa\_{p,d} \mathcal{D}\_p(\mu\_N, \mu) \\ &\leq \kappa\_{p,d} \sum\_{n\geq 0} 2^{pn} |\mu\_N(B\_n) - \mu(B\_n)| \\ &\quad + \kappa\_{p,d} \sum\_{n\geq 0} 2^{pn} \mu(B\_n) \mathcal{D}\_p(\mathcal{R}\_{B\_n} \mu\_N, \mathcal{R}\_{B\_n} \mu) \\ &=: \kappa\_{p,d} (\mathcal{Z}\_N^p + V\_N^p). \end{split}$$

Hence

$$\Pr(\mathcal{T}\_p(\mu\_N, \mu) \ge \ge) \le \Pr(Z\_N^p \ge \ge/(2\kappa\_{p,d})) + \Pr(V\_N^p \ge \ge/(2\kappa\_{p,d})).$$

By Lemma [13](#page-18-0) (choosing *<sup>x</sup>*<sup>0</sup> := <sup>1</sup>/(2κ*p*,*<sup>d</sup>* )), we easily find Pr(*<sup>Z</sup> <sup>p</sup> <sup>N</sup>* ≥ *x*/(2κ*p*,*<sup>d</sup>* )) ≤ *Ce*−*cNx*<sup>2</sup> **1**{*x*≤1} + *b*(*N*, *x*) ≤ *a*(*N*, *x*)**1**{*x*≤1} + *b*(*N*, *x*), these quantities being defined in the statement of Theorem [2.](#page-3-1) We now check that there is *A* > 0 such that for all *x* > 0,

<span id="page-24-0"></span>
$$\Pr\left[V\_N^p \ge \ge / (2\kappa\_{p,d})\right] \le a(N, \ge) \mathbf{1}\_{\{\underline{\mathbf{x}} \le A\}}.\tag{6}$$

This will end the proof, since one easily checks that *a*(*N*, *x*)**1**{*x*≤*A*} ≤ *a*(*N*, *x*)**1**{*x*≤1}+ *b*(*N*, *x*) (when allowing the values of the constants to change).

Let us thus check [\(6\)](#page-24-0). For η > 0 to be chosen later, we set *<sup>c</sup>* := (<sup>1</sup> <sup>−</sup> <sup>2</sup>−η)/(2κ*p*,*<sup>d</sup>* ) and *zn* := *cx*2−(*p*+η)*n*/μ(*Bn*). Observing that *<sup>n</sup>*≥0(<sup>1</sup> <sup>−</sup> <sup>2</sup>−η)2−η*<sup>n</sup>* <sup>=</sup> 1), we write

$$\begin{aligned} \mathbb{P}\left(V\_N^p \ge x/(2\kappa\_{p,d})\right) &\le \left(\sum\_{n\ge 0} \mathbb{P}\left[\mathcal{D}\_p(\mathcal{R}\_{B\_n}\mu\_N, \mathcal{R}\_{B\_n}\mu) \ge z\_n\right]\right) \wedge 1\\ &=: \left(\sum\_{n\ge 0} K\_n(N, x)\right) \wedge 1. \end{aligned}$$

From now on, the value of *c* > 0 is not allowed to vary anymore. We introduce another positive constant *a* > 0 whose value may change from line to line. We only assume [\(3\)](#page-3-2) (which is implied by [\(1\)](#page-3-3) or [\(2\)](#page-3-0)). We now show that if η > 0 is chosen small enough, there is *A* > 0 such that

<span id="page-24-1"></span>
$$\sum\_{n\geq 0} K\_n(N, \mathbf{x}) \leq C \exp(-aNh(\mathbf{x})) \mathbf{1}\_{\{\mathbf{x} \leq A\}} \quad \text{if} \ Nh(\mathbf{x}) \geq 1,\tag{7}$$

where *<sup>h</sup>*(*x*) <sup>=</sup> *<sup>x</sup>*<sup>2</sup> if *<sup>p</sup>* <sup>&</sup>gt; *<sup>d</sup>*/2, *<sup>h</sup>*(*x*) <sup>=</sup> (*x*/ log(2+1/*x*))<sup>2</sup> if *<sup>p</sup>* <sup>=</sup> *<sup>d</sup>*/2 and *<sup>h</sup>*(*x*) <sup>=</sup> *<sup>x</sup>d*/*<sup>p</sup>* if *p* < *d*/2. This will obviously imply as usual that for all *x* > 0,

$$\left(\sum\_{n\geq 0} K\_n(N,\boldsymbol{x})\right) \wedge 1 \leq C \exp(-aNh(\boldsymbol{x})) \mathbf{1}\_{\{\boldsymbol{x}\leq A\}}$$

and thus conclude the proof of [\(6\)](#page-24-0). We thus only have to prove [\(7\)](#page-24-1).

Conditionally on μ*<sup>N</sup>* (*Bn*), *RBn*μ*<sup>N</sup>* is the empirical measure of *N*μ*<sup>N</sup>* (*Bn*) points which are *RBn*μ-distributed. Since *RBn*μ is supported in (−1, 1] *<sup>d</sup>* , we may apply Proposition [10](#page-15-2) and obtain

$$\begin{aligned} K\_n(N, \boldsymbol{x}) &\leq C \mathbb{E} \left[ \mathbf{1}\_{\{\boldsymbol{z}\_n \leq 1\}} \exp \left( -a N \mu\_N(B\_n) h(\boldsymbol{z}\_n) \right) \right] \\ &\leq C \mathbf{1}\_{\{\boldsymbol{z}\_n \leq 1\}} \exp(-N \mu(B\_n) (1 - e^{-ah(\boldsymbol{z}\_n)})) \end{aligned}$$

by Lemma [12-](#page-17-1)(c). But the condition *zn* ≤ 1 implies that *h*(*zn*) is bounded (by a constant depending only on *p* and *d*), whence

$$K\_n(N, \chi) \le C \mathbf{1}\_{\{\varepsilon\_n \le 1\}} \exp(-aN\mu(B\_n)h(\varepsilon\_n)).$$

By [\(3\)](#page-3-2), we have μ(*Bn*) <sup>≤</sup> *<sup>C</sup>*2−*qn*. Hence if *<sup>x</sup>* <sup>&</sup>gt; *<sup>A</sup>* := *<sup>C</sup>*/*c*, we have *zn* <sup>≥</sup> (*c*/*C*)*x*2(*q*−*p*−η)*<sup>n</sup>* <sup>&</sup>gt; 1 for all *<sup>n</sup>* <sup>≥</sup> 1 (if <sup>η</sup> <sup>∈</sup> (0, *<sup>q</sup>* <sup>−</sup> *<sup>p</sup>*)) and thus *<sup>n</sup>*≥<sup>0</sup> *Kn*(*N*, *<sup>x</sup>*) <sup>=</sup> <sup>0</sup> as desired.

Next, we see that θ → θ*h*(*x*/θ ) is decreasing, whence for all *x* ≤ *A*,

$$K\_n(N, \mathbf{x}) \le C \exp(-aN2^{-qn}h(c\ge 2^{(q-p-\eta)n}/C)) \le C \exp(-aN2^{-qn}h(\ge 2^{(q-p-\eta)n})).$$

We now treat separately the three cases.

**Step 1: case** *<sup>p</sup>* <sup>&</sup>gt; *<sup>d</sup>*/2. Since *<sup>h</sup>*(*x*) <sup>=</sup> *<sup>x</sup>*2, we have, if <sup>η</sup> <sup>∈</sup> (0, *<sup>q</sup>*/<sup>2</sup> <sup>−</sup> *<sup>p</sup>*),

$$\sum\_{n\geq 0} K\_n(N, \mathbf{x}) \leq C \sum\_{n\geq 0} \exp\left(-aN\mathbf{x}^2 2^{n(q-2p-2\eta)}\right) \leq C \exp(-aN\mathbf{x}^2)$$

if *N x*<sup>2</sup> <sup>≥</sup> 1.

**Step 2: case** *<sup>p</sup>* <sup>=</sup> *<sup>d</sup>*/2. Since *<sup>h</sup>*(*x*) <sup>=</sup> (*x*/ log(2+1/*x*))2, we have, if <sup>η</sup> <sup>∈</sup> (0,q/2-p),

$$\begin{aligned} \sum\_{n\geq 0} K\_{\mathbb{H}}(N, x) &\leq C \sum\_{n\geq 0} \exp\left(-aNx^2 2^{(q-2p-2\eta)n} / \log^2(2 + 1/(x2^{(q-p-\eta)n})) \right) \\ &\leq C \sum\_{n\geq 0} \exp(-aNh(\mathbf{x})2^{n(q-2p-2\eta)}) \\ &\leq C \exp(-aNh(\mathbf{x})) \end{aligned}$$

if *N h*(*x*) <sup>≥</sup> 1. The third inequality only uses that log2(2+1/(*x*2*n*(*q*−*p*−η))) <sup>≤</sup> log2(2<sup>+</sup> 1/*x*).

**Step 3: case** *<sup>p</sup>* <sup>&</sup>lt; *<sup>d</sup>*/2. Here *<sup>h</sup>*(*x*) <sup>=</sup> *<sup>x</sup>d*/*p*. Since *<sup>p</sup>* <sup>&</sup>lt; *<sup>d</sup>*/2 and *<sup>q</sup>* <sup>&</sup>gt; <sup>2</sup>*p*, it holds that *q*(1 − *p*/*d*) − *p* > 0. We thus may take η ∈ (0, *q*(1 − *p*/*d*) − *p*) (so that *q*(*d*/*p* − 1) − *d* − *d*η/*p* > 0) and we get

$$\sum\_{n\geq 0} K\_n(N, \boldsymbol{x}) \leq C \sum\_{n\geq 0} \exp(-aN\boldsymbol{x}^{d/p}2^{n(q(d/p-1)-d-d\eta/p)}) \leq C \exp(-aN\boldsymbol{x}^{d/p})$$

if *N xd*/*<sup>p</sup>* <sup>≥</sup> 1.

#### <span id="page-25-0"></span>**7 The dependent case**

We finally study a few classes of dependent sequences of random variables. We only give some moment estimates. Concentration inequalities might be obtained, but this should be much more complicated.

## 7.1 ρ-mixing stationary sequences

A stationary sequence of random variables (*Xn*)*n*≥<sup>1</sup> with common law μ is said to be <sup>ρ</sup>-mixing, for some <sup>ρ</sup> : <sup>N</sup> <sup>→</sup> <sup>R</sup><sup>+</sup> with <sup>ρ</sup>*<sup>n</sup>* <sup>→</sup> 0, if for all *<sup>f</sup>*, *<sup>g</sup>* <sup>∈</sup> *<sup>L</sup>*2(μ) and all *<sup>i</sup>*, *<sup>j</sup>* <sup>≥</sup> <sup>1</sup>

$$\mathbb{Cov}\left(f(X\_{l}),g(X\_{j})\right) \leq \rho\_{|l-j|}\sqrt{\mathbb{Var}\left(f(X\_{l})\right)\mathbb{Var}\left(g(X\_{j})\right)}.$$

We refer for example to Rio [\[37](#page-31-10)], Doukhan [\[20\]](#page-30-17) or Bradley [\[10\]](#page-30-20).

**Theorem 14** *Consider a stationary sequence of random variables*(*Xn*)*n*≥<sup>1</sup> *with common law* <sup>μ</sup> *and set* <sup>μ</sup>*<sup>N</sup>* := *<sup>N</sup>*−<sup>1</sup> *<sup>N</sup>* <sup>1</sup> δ*Xi . Assume that this sequence is* ρ*-mixing, for some* <sup>ρ</sup> : <sup>N</sup> <sup>→</sup> <sup>R</sup><sup>+</sup> *satisfying <sup>n</sup>*≥<sup>0</sup> <sup>ρ</sup>*<sup>n</sup>* <sup>&</sup>lt; <sup>∞</sup>*. Let p* <sup>&</sup>gt; <sup>0</sup> *and assume that* <sup>μ</sup> <sup>∈</sup> *Mq* (R*<sup>d</sup>* ) *for some p* > *q. There exists a constant C depending only on p*, *d*, *q*, *Mq* (μ), ρ *such that, for all N* ≥ 1*,*

$$\mathbb{E}\left(\mathcal{T}\_{\mathcal{P}}(\mu\_N,\mu)\right) \leq C \begin{cases} N^{-1/2} + N^{-(q-p)/q} & \text{if } p > d/2 \quad \text{and} \quad q \neq 2p, \\ N^{-1/2}\log(1+N) + N^{-(q-p)/q} & \text{if } p = d/2 \quad \text{and} \quad q \neq 2p, \\ N^{-p/d} + N^{-(q-p)/q} & \text{if } p \in (0,d/2) \quad \text{and} \quad q \neq d/(d-p). \end{cases}$$

 This is very satisfying: we get the same estimate as in the independent case. The case *<sup>n</sup>*≥<sup>0</sup> <sup>ρ</sup>*<sup>n</sup>* = ∞ can also be treated (but then the upper bounds will be less good and depend on the rate of decrease of ρ). Actually, the ρ-mixing condition is slightly too strong (we only need the covariance inequality when *f* = *g* is an indicator function), but it is best adapted notion of mixing we found in the litterature.

*Proof* We first check that for any Borel subset *<sup>A</sup>* <sup>⊂</sup> <sup>R</sup>*<sup>d</sup>* ,

$$\mathbb{E}[|\mu\_N(A) - \mu(A)|] \le \min\{2\mu(A), C\mu(A)N^{-1/2}\}.$$

But this is immediate: <sup>E</sup>[μ*<sup>N</sup>* (*A*)] = μ(*A*) (whence <sup>E</sup>[|μ*<sup>N</sup>* (*A*) <sup>−</sup> μ(*A*)|] ≤ <sup>2</sup>μ(*A*)) and

$$\begin{aligned} \mathbb{V}\text{ar }\mu\_N(A) &= \frac{1}{N^2} \sum\_{i,j \le N} \mathbb{C} \text{ov}\left(\mathbf{1}\_A(X\_i), \mathbf{1}\_A(X\_i)\right) \\ &\le \frac{1}{N^2} \sum\_{i,j \le N} \rho\_{|i-j|} \mathbb{V} \text{ar}\left(\mathbf{1}\_A(X\_1)\right) \\ &\le \frac{\mu(A)(1-\mu(A))}{N^2} \sum\_{i,j \le N} \rho\_{|i-j|}. \end{aligned}$$

This is smaller than *C*μ(*A*)/*N* as desired, since *<sup>i</sup>*,*j*≤*<sup>N</sup>* <sup>ρ</sup>|*i*<sup>−</sup> *<sup>j</sup>*<sup>|</sup> <sup>≤</sup> *<sup>N</sup> <sup>k</sup>*≥<sup>0</sup> <sup>ρ</sup>*<sup>k</sup>* <sup>=</sup> *C N*. Once this is done, it suffices to copy (without any changes) the proof of Theorem [1.](#page-1-0) 

# 7.2 Markov chains

Here we consider a <sup>R</sup>*<sup>d</sup>* -valued Markov chain (*Xn*)*n*≥<sup>1</sup> with transition kernel *<sup>P</sup>* and initial distribution <sup>ν</sup> <sup>∈</sup> *<sup>P</sup>*(R*<sup>d</sup>* ) and we set <sup>μ</sup>*<sup>N</sup>* := *<sup>N</sup>*−<sup>1</sup> *<sup>N</sup>* <sup>1</sup> δ*Xn* . We assume that it admits a unique invariant probability measure π and the following *L*2-decay property (usually related to a Poincaré inequality)

<span id="page-27-0"></span>
$$\forall \, n \ge 1, \,\,\forall \, f \in L^2(\pi), \quad \|P^n f - \pi(f)\|\_{L^2(\pi)} \le \rho\_n \|f - \pi(f)\|\_{L^2(\pi)}\tag{8}$$

for some sequence ρ = (ρ*n*)*n*≥<sup>1</sup> decreasing to 0.

**Theorem 15** *Let p* ≥ 1*, d* ≥ 1 *and r* > 2 *be fixed. Assume that our Markov chain* (*Xn*)*n*≥<sup>0</sup> *satisfies* [\(8\)](#page-27-0) *with a sequence* (ρ*n*)*n*≥<sup>1</sup> *satisfying <sup>n</sup>*≥<sup>1</sup> <sup>ρ</sup>*<sup>n</sup>* <sup>&</sup>lt; <sup>∞</sup>*. Assume also that the initial distribution* ν *is absolutely continuous with respect to* π *and satisfies d*ν/*d*π*Lr*(π ) < ∞*. Assume finally that Mq* (π ) < ∞ *for some q* > *pr*/(*r* − 1)*. Setting qr* := *q*(*r* − 1)/*r and dr* = *d*(*r* + 1)/*r, there is a constant C, depending only on p*, *d*,*r*, *q*,ρ, *Mq* (π ) *and d*ν/*d*π*Lr*(π ) *such that for all N* ≥ 1*,*

$$\mathbb{E}\_{\nu}\left(\mathcal{T}\_{p}(\mu\_{N},\pi)\right) \leq C \begin{cases} N^{-1/2} + N^{-(q\_{r}-p)/q\_{r}} & \text{if } p > d\_{r}/2r \quad \text{and} \quad q\_{r} \neq 2p, \\ N^{-1/2}\log(1+N) + N^{-(q\_{r}-p)/q\_{r}} & \text{if } p = d\_{r}/2r \quad \text{and} \quad q\_{r} \neq 2p, \\ N^{-p/d} + N^{-(q\_{r}-p)/q\_{r}} & \text{if } p \in (0,d\_{r}/2) \quad \text{and} \quad q\_{r} \neq d\_{r}/(d\_{r}-p). \end{cases}$$

Once again, we might adapt the proof to get a complete picture corresponding to other decay than *<sup>L</sup>*2-*L*<sup>2</sup> and to slower mixing rates (ρ*n*)*n*≥1.

*Proof* We only have to show that for any ≥ 0, any *n* ≥ 0,

$$\begin{split} \Delta\_{n,\ell}^{N} &:= \sum\_{F \in \mathcal{P}\_{\ell}} \mathbb{E}\_{\nu} \left( |\mu\_{N}(\mathcal{Q}^{n}F \cap B\_{n}) - \pi(\mathcal{Q}^{n}F \cap B\_{n})| \right) \\ &\leq C \min \left\{ (\pi(B\_{n}))^{(r-1)/r}, \, [2^{d\_{\ell}\ell}(\pi(B\_{n}))^{(r-1)/r}/N]^{1/2} \right\}. \end{split}$$

Since *Mq* (π ) < <sup>∞</sup> (whence π(*Bn*) <sup>≤</sup> *<sup>C</sup>*2−*qn*), we will deduce that

$$
\Delta\_{n,\ell}^N \le C \min\left\{ 2^{-q\_r n}, 2^{d\_r \ell/2} (2^{-q\_r n}/N)^{1/2} \right\}.
$$

Then the rest of the proof is exactly the same as that of Theorem [1,](#page-1-0) replacing everywhere *q* and *d* by *qr* and *dr*.

We first check that *<sup>N</sup> <sup>n</sup>*, <sup>≤</sup> *<sup>C</sup>*(π(*Bn*))(*r*−1)/*r*. Using that *d*ν/*d*π*Lr*(π ) <sup>&</sup>lt; <sup>∞</sup>, we write

$$\mathbb{E}\_{\boldsymbol{\nu}}(\mu\_N(B\_n)) = \frac{1}{N} \sum\_{l=1}^N \mathbb{E}\_{\boldsymbol{\pi}} \left[ \frac{d\boldsymbol{\nu}}{d\pi}(X\_0) \mathbf{1}\_{\{X\_l \in B\_n\}} \right] \le \|d\boldsymbol{\nu}/d\pi\|\_{L^r(\boldsymbol{\pi})} \pi(B\_n)^{(r-1)/r}.$$

We next consider a Borel subset *A* of R*<sup>d</sup>* and check that

$$\mathbb{E}\_{\boldsymbol{\nu}}(|\mu\_N(A) - \boldsymbol{\pi}(A)|) \le C(\boldsymbol{\pi}(A))^{(r-1)/(2r)} N^{-1/2}.$$

To do so, as is usual when working with Markov chains or covariance properties (see [\[7](#page-30-15)]), we introduce *f* = 1*<sup>A</sup>* − π(*A*) and write

$$\mathbb{E}\_{\boldsymbol{\nu}}(|\mu\_N(A) - \boldsymbol{\pi}(A)|) = \frac{1}{N} \mathbb{E}\_{\boldsymbol{\nu}}\left(\left|\sum\_{i=1}^N f(X\_i)\right|\right) \le \frac{1}{N} \left(\sum\_{i,j=1}^N \mathbb{E}\_{\boldsymbol{\nu}}(f(X\_i)f(X\_j))\right)^{1/2}.$$

For *j* ≥ *i*, it holds that

$$\mathbb{E}\_{\mathbb{V}}(f(X\_l)f(X\_j)) = \mathbb{E}\_{\mathbb{V}}[f(X\_l)P^{j-l}f(X\_l)] = \mathbb{E}\_{\pi}\left[\frac{d\upsilon}{d\pi}(X\_0)f(X\_l).P^{j-l}f(X\_l)\right].$$

Using the Hölder inequality (recall that *d*ν/*d*π*Lr*(π ) < ∞ with *r* > 2) and [\(8\)](#page-27-0), we get

$$\begin{aligned} \mathbb{E}\_{\nu}(f(X\_{l})f(X\_{j})) &\leq \|d\nu/d\pi\|\_{L^{r}(\mathfrak{x})} \|f\|\_{L^{2r/(r-2)}(\mathfrak{x})} \|P^{j-l}f\|\_{L^{2}(\mathfrak{x})} \\ &\leq \mathsf{C}\rho\_{j-l} \|f\|\_{L^{2r/(r-2)}(\mathfrak{x})} \|f\|\_{L^{2}(\mathfrak{x})}. \end{aligned}$$

But for *<sup>s</sup>* <sup>&</sup>gt; 1, *<sup>f</sup> Ls*(π ) <sup>≤</sup> *Cs*(π(*A*) <sup>+</sup> (π(*A*))*s*)1/*<sup>s</sup>* <sup>≤</sup> *Cs*(π(*A*))1/*s*, we find <sup>E</sup><sup>ν</sup> ( *<sup>f</sup>* (*Xi*) *<sup>f</sup>* (*<sup>X</sup> <sup>j</sup>*)) <sup>≤</sup> *<sup>C</sup>*ρ*j*−*i*(π(*A*))(*r*−1)/*<sup>r</sup>* and thus

$$\begin{aligned} \mathbb{E}\_{\boldsymbol{\nu}}(|\mu\_N(F) - \pi(F)|) &\leq \frac{C}{N} \left( \sum\_{l,j=1}^N \rho\_{|l-j|} (\pi(F))^{(r-1)/2r} \right)^{1/2} \\ &\leq C (\pi(F))^{(r-1)/(2r)} N^{-1/2} \end{aligned}$$

as desired. We used that *<sup>N</sup> <sup>i</sup>*,*j*=<sup>1</sup> <sup>ρ</sup>|*i*<sup>−</sup> *<sup>j</sup>*<sup>|</sup> <sup>≤</sup> *C N*. We can finally conclude that

$$\Delta\_{n,\ell}^{N} \le CN^{-1/2} \sum\_{F \in \mathcal{P}\_{\ell}} (\pi(2^n F \cap B\_n))^{(r-1)/(2r)} \le CN^{-1/2} 2^{d\_r \ell/2} (\pi(B\_n))^{(r-1)/(2r)}$$

by the Hölder inequality (and because #*<sup>P</sup>* <sup>=</sup> <sup>2</sup>*<sup>d</sup>*), where *dr* <sup>=</sup> *<sup>d</sup>*(*<sup>r</sup>* <sup>+</sup> <sup>1</sup>)/*<sup>r</sup>* as in the statement.

#### 7.3 Mc Kean-Vlasov particles systems

Particle approximation of nonlinear equations has attracted a lot of attention in the past thirty years. We will focus here on the following R*<sup>d</sup>* -valued nonlinear S.D.E.

$$dX\_t = \sqrt{2}dB\_t - \nabla V(X\_t)dt - \nabla W \* u\_t(X\_t)dt, \qquad X\_0 = x\_0$$

where *ut* <sup>=</sup> *La*w(*Xt*) and (*Bt*)is an <sup>R</sup>*<sup>d</sup>* -valued Brownian motion. This is a probabilistic representation of the so-called Mc Kean-Vlasov equation, which has been studied in particular by Carillo et al. [\[12](#page-30-21)], Malrieu [\[28\]](#page-30-22) and Cattiaux et al.[\[13](#page-30-23)] to which we refer for further motivations and existence and uniqueness of solutions. We will mainly consider here the case where *V* and *W* are convex (and if *V* = 0 the center of mass is fixed) and *W* is even. To fix the ideas, let us consider only two cases:

- (a) *Hess V* ≥ β*I d* > 0, *Hess W* ≥ 0.
- (b) *V*(*x*) = |*x*| <sup>α</sup> for α > 2, *Hess W* <sup>≥</sup> 0.

The particle system introduced to approximate the nonlinear equation is the following. Let(*B<sup>i</sup> <sup>t</sup>* )*t*≥<sup>0</sup> be *<sup>N</sup>* independent Brownian motions. For*<sup>i</sup>* <sup>=</sup> <sup>1</sup>,..., *<sup>N</sup>*, set *<sup>X</sup>i*,*<sup>N</sup>* <sup>0</sup> = *x* and

$$dX\_{l}^{l,N} = \sqrt{2}dB\_{l}^{l} - \nabla V(X\_{l}^{l,N})dt - \frac{1}{N} \sum\_{j} \nabla W(X\_{l}^{l,N} - X\_{l}^{j,N})dt.$$

Usual propagation of chaos property is usually concerned with control of

$$\mathcal{T}\_2(Law(X\_1^{1,N}), u\_I)$$

uniformly (or not) in time. It is however very natural to consider rather a control of

*<sup>T</sup>*2(*u*<sup>ˆ</sup> *<sup>N</sup> <sup>t</sup>* , *ut*)

where *<sup>u</sup>*<sup>ˆ</sup> *<sup>N</sup> <sup>t</sup>* <sup>=</sup> <sup>1</sup> *N <sup>N</sup> <sup>i</sup>*=<sup>1</sup> <sup>δ</sup>*Xi*,*<sup>N</sup> <sup>t</sup>* , as in Bolley et al. [\[11](#page-30-12)].

To do so, and inspired by the usual proof of propagation of chaos, let us consider nonlinear independent particles

$$dX\_t^l = \sqrt{2}dB\_t^l - \nabla V(X\_t^l)dt - \nabla W \* \mu\_l(X\_t^l)dt, \qquad X\_0^l = x\_0^l$$

(driven by the same Brownian motions as the particle system) and the corresponding empirical measure *u<sup>N</sup> <sup>t</sup>* <sup>=</sup> <sup>1</sup> *N <sup>N</sup> <sup>i</sup>*=<sup>1</sup> <sup>δ</sup>*X<sup>i</sup> t* . We then have

$$\mathcal{T}\_2\left(\hat{\mu}\_l^N, \mu\_l\right) \le 2\mathcal{T}\_2\left(\hat{\mu}\_l^N, \mu\_l^N\right) + 2\mathcal{T}\_2\left(\mu\_l^N, \mu\_l\right).$$

Then following [\[28\]](#page-30-22) in case (a) and [\[13\]](#page-30-23) in case (b), one easily gets (for some timeindependent constant *C*)

$$\mathbb{E}\left(T\_2^2(\hat{u}\_t^N, u\_t^N)\right) \le \frac{1}{N} \mathbb{E}\left(\sum\_{l=1}^N |X\_l^{l,N} - X\_l^l|^2\right) \le C\alpha(N).$$

where α(*n*) <sup>=</sup> *<sup>N</sup>*−<sup>1</sup> in case (a) and α(*N*) <sup>=</sup> *<sup>N</sup>*−1/(α−1) in case (b). It is not hard to prove here that the nonlinear particles have infinitely many moments (uniformly in time) so that combining Theorem [1](#page-1-0) with the previous estimates gives

$$\sup\_{t \ge 0} \mathbb{E}(T\_2(\hat{\mu}\_l^N, \mu\_l)) \le C(\alpha(N) + \beta(N))$$

where β(*N*) <sup>=</sup> *<sup>N</sup>*−1/<sup>2</sup> if *<sup>d</sup>* <sup>=</sup> 1, β(*N*) <sup>=</sup> *<sup>N</sup>*−1/<sup>2</sup> log(<sup>1</sup> <sup>+</sup> *<sup>N</sup>*) if *<sup>d</sup>* <sup>=</sup> 2 and β(*N*) <sup>=</sup> *<sup>N</sup>*−1/*<sup>d</sup>* if *<sup>d</sup>* <sup>≥</sup> 3.

# **References**

- 1. Adamczak, R.: A tail inequality for suprema of unbounded empirical processes with applications to Markov chains. Electron. J. Probab. **13**, 1000–1034 (2008)
- <span id="page-30-1"></span>2. Ajtai, M., Komlós, J., Tusnády, G.: On optimal matchings. Combinatorica **4**, 259–264 (1984)
- <span id="page-30-2"></span>3. Barthe, F., Bordenave, C.: Combinatorial optimization over two random point sets. Séminaire de Probabilités XLV, Lecture Notes in Mathematics 2078, pp. 483–535, Springer, Berlin (2013)
- <span id="page-30-18"></span>4. Bennett, G.: Probability inequalities for the sum of independent random variables. J. Am. Statist. Assoc. **57**, 33–45 (1962)
- <span id="page-30-10"></span>5. Biau, G., Devroye, L., Lugosi, G.: On the performance of clustering in Hilbert spaces. IEEE Trans. Inf. Theory **54**, 781–790 (2008)
- <span id="page-30-16"></span>6. Boissard, E.: Simple bounds for the convergence of empirical and occupation measures in 1-Wasserstein distance. Electron. J. Probab. **16**, 2296–2333 (2011)
- <span id="page-30-15"></span>7. Boissard, E., Le Gouic, T.: On the mean speed of convergence of empirical and occupation measures in Wasserstein distance. [arXiv:1105.5263](http://arxiv.org/abs/1105.5263)
- <span id="page-30-6"></span>8. Borovkov, A.A.: Estimates for the distribution of sums and maxima of sums of random variables when the Cramér condition is not satisfied. Sib. Math. J. **41**, 811–848 (2000)
- 9. Bradley, R.C.: A central limit theorem for stationary ρ-mixing sequences with infinite variance. Ann. Probab. **16**, 313–332 (1988)
- <span id="page-30-20"></span>10. Bradly, R.C.: Introduction to Strong Mixing Conditions, vol. 1,2,3. Kendrick Press, Heber City (2007)
- <span id="page-30-12"></span>11. Bolley, F., Guillin, A., Villani, C.: Quantitative concentration inequalities for empirical measures on non-compact spaces. Probab. Theory Relat. Fields **137**, 541–593 (2007)
- <span id="page-30-21"></span>12. Carrillo, J.-A., Mac Cann, R., Villani, C.: Kinetic equilibration rates for granular media and related equations: entropy dissipation and mass transportation. Rev. Mat. Iberoam. **19**, 971–1018 (2003)
- <span id="page-30-23"></span>13. Cattiaux, P., Guillin, A., Malrieu, F.: Probabilistic approach for granular media equations in the non uniformly convex case. Probab. Theory Relat. Fields **140**, 19–40 (2008)
- <span id="page-30-8"></span>14. Delattre, S., Graf, S., Luschgy, H., Pagès, G.: Quantization of probability distributions under normbased distortion measures. Stat. Decis. **22**, 261–282 (2004)
- 15. Dereich, S.: Asymptotic formulae for coding problems and intermediate optimization problems: a review. In: Trends in Stochastic Analysis. pp. 187–232, Cambridge University Press, Cambridge (2009)
- <span id="page-30-0"></span>16. Dereich, S., Scheutzow, M., Schottstedt, R.: Constructive quantization: approximation by empirical measures. Ann. Inst. Henri Poincar Probab. Stat. **49**, 1183–1203 (2013)
- <span id="page-30-19"></span>17. Devroye, L., Lugosi, G.: Combinatorial Methods in Density Estimation. Springer, Berlin (2001)
- <span id="page-30-3"></span>18. Djellout, H., Guillin, A., Wu, L.: Transportation cost-information inequalities and applications ro random dynamical systems and diffusions. Ann. Probab. **32**, 2702–2732 (2004)
- <span id="page-30-9"></span>19. Dobri´c, V., Yukich, J.E.: Asymptotics for transportation cost in high dimensions. J. Theor. Probab. **8**, 97–118 (1995)
- <span id="page-30-17"></span>20. Doukhan, P.: Mixing: Properties and Examples. Springer, New-York (1995)
- 21. Dudley, R.M.: Central limit theorems for empirical measures. Ann. Probab. **6**, 899–929 (1978)
- <span id="page-30-13"></span>22. Fournier, N., Mischler, S.: Rate of convergence of the Nanbu particle system for hard potentials. [arXiv:1302.5810](http://arxiv.org/abs/1302.5810)
- <span id="page-30-7"></span>23. Fuk, D.H., Nagaev, S.V.: Probability inequalities for sums of independent random variables. Theory Probab. Appl. **16**, 660–675 (1971)
- <span id="page-30-4"></span>24. Gozlan, N.: Integral criteria for transportation cost inequalities. Electron. Commun. Probab. **11**, 64–77 (2006)
- <span id="page-30-14"></span>25. Horowitz, J., Karandikar, R.L.: Mean rates of convergence of empirical measures in the Wasserstein metric. J. Comput. Appl. Math. **55**, 261–273 (1994)
- <span id="page-30-11"></span>26. Laloë, T.: *L*1-Quantization and clustering in Banach spaces. Math. Method Stat. **19**, 136–150 (2009)
- <span id="page-30-5"></span>27. Ledoux, M.: The Concentration of Measure Phenomenon. Mathematical Surveys and Monographs 89. American Math. Society, Providence (2001)
- <span id="page-30-22"></span>28. Malrieu, F.: Convergence to equlibrium for granular media equations. Ann. Appl. Probab. **13**, 540–560 (2003)
- 29. Massart, P.: Concentration Inequalities and Model Selection: Ecole d'Été de Probabilités de Saint-Flour XXXIII. Springer, Berlin (2003)
- 30. Merlevède, F., Peligrad, M.: Rosenthal-type inequalities for the maximum of partial sums of stationary processes and examples. Ann. Probab. **41**, 914–960 (2013)
- <span id="page-31-4"></span>31. Merlevède, F., Peligrad, M., Rio, E.: A Bernstein type inequality and moderate deviations for weakly dependent sequences. Probab. Theory Relat. Fields **151**, 435–474 (2011)
- <span id="page-31-8"></span>32. Mischler, S., Mouhot, C.: Kac's programm in kinetic theory. Invent. Math. **193**, 1–147 (2013)
- <span id="page-31-5"></span>33. Pagès, G., Wilbertz, B.: Optimal Delaunay and Voronoi quantization schemes for pricing American style options. In: Carmona, R., Hu, P., Del Moral, P., Oudjane, N. (eds.) Numerical Methods in Finance, pp. 171–217. Springer, Berlin (2012)
- <span id="page-31-9"></span>34. Perrin, D.: Une variante de la formule de Stirling. [http://www.math.u-psud.fr/~perrin/CAPES/analyse/](http://www.math.u-psud.fr/~perrin/CAPES/analyse/Suites/Stirling) [Suites/Stirling](http://www.math.u-psud.fr/~perrin/CAPES/analyse/Suites/Stirling)
- <span id="page-31-7"></span>35. Rachev, S.T., Rüschendorf, L.: Mass Transportation Problems. Vol. I. and II. Probability and its Applications. Springer, Berlin (1998)
- <span id="page-31-6"></span>36. Roberts, G., Rosenthal, J.-S.: Shift-coupling and convergence rates of ergodic averages. Commun. Stat. Stoch. Models **13**, 147–165 (1996)
- <span id="page-31-10"></span>37. Rio, E.: Théorie asymptotique des processus aléatoires faiblement dépendants, Mathématiques et Applications 31. Springer, Paris (2000)
- <span id="page-31-2"></span>38. Talagrand, M.: Matching random samples in many dimensions. Ann. Appl. Probab. **2**, 846–856 (1992)
- <span id="page-31-3"></span>39. Talagrand, M.: The transportation cost from the uniform measure to the empirical measure in dimension ≥ 3. Ann. Probab. **22**, 919–959 (1994)
- <span id="page-31-0"></span>40. Van der Vaart, A., Wellner, J.A.: Weak Convergence of Empirical Processes. Springer, Berlin (1996)
- <span id="page-31-1"></span>41. Villani, C.: Topics in Optimal Transportation. Graduate Studies in Mathematics, 58. American Mathematical Society, Providence (2003)