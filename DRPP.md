# Distributionally Robust Probabilistic Prediction for Stochastic Dynamical Systems

Tao Xu and Jianping He

Abstract-Probabilistic prediction of stochastic dynamical systems (SDSs) aims to accurately predict the conditional probability distributions of future states.However,accurate probabilistic predictions tightly hinge on accurate distributional information from a nominal model,which is hardly available in practice.To address this issue,we propose a novel functional-maximin-based distributionally robust probabilistic prediction (DRPP） framework.In this framework,one can design probabilistic predictors that have worst-case performance guarantees over a pre-defined ambiguity set of SDSs. Nevertheless, DRPP requires optimizing over the space of probability measures with density functions with respect to the Lebesgue measure,which is generally intractable. We develop a methodology that equivalently transforms the original maximin from function spaces to Euclidean spaces. Although it remains intractable to seek a global optimal solution, two suboptimal solutions are derived.By relaxing the constraints on the ambiguity set,we obtain a suboptimal predictor called Noise-DRPP. Relaxing the constraints on the predictor yields another suboptimal predictor, Eig-DRPP. Moreover, optimality gaps between the proposed predictors and the global optimal predictor are derived.Finally,we conduct elaborate numerical simulations to compare the performance of different predictors under different SDSs.

Index Terms-Stochastic Dynamical System,Distributionally Robust Optimization,Uncertainty Quantification,Probabilistic Prediction.

# I. INTRODUCTION

# A. Background

A probabilistic prediction is a forecast that assigns probabilities to different possible outcomes,enabling better uncertainty quantification,risk assessment,and decision-making.Due to the ever-increasing demand for both accurate and reliable predictions against uncertainty [1], it has wide applications in fields such as epidemiology [2],climatology [3],and robotics [4]，etc.The performance of a probabilistic prediction is evaluated by both sharpness and calibration of the predictive probability density function (pdf). The sharpness reflects how concentrated (e.g.,low-variance or low-entropy) the predictive pdf is,subject to the intrinsic stochasticity of the underlying system. The calibration concerns the statistical compatibility between the predictive pdf and the realized outcomes [5]. To simultaneously assess both sharpness and calibration, one usually assigns a numerical score to the predictive pdf and the realized value of the prediction target. If one has a further requirement that the expected scoring rule is maximized only when the predictive pdf equals the target's real pdf almost everywhere,a strictly proper scoring rule (see in [6]) is needed.

The probabilistic prediction of a stochastic dynamical system (SDS） is to predict the conditional pdfs of future states based on some prior information about the system. One's prior information is usually in the form of a nominal model, which is a simplified or idealized representation of the underlying system.While probabilistic prediction for a linear system with Gaussian noises is straightforward, it becomes computationally expensive once the nominal SDS is nonlinear or the noises are non-Gaussian. Therefore,a lot of research contributes to approximating the future pdf to balance the requirement of prediction precision and computation complexity [7].

# B. Motivation

Despite the significant achievements in the current prediction algorithms for SDSs,they are not guaranteed to perform well if the distributional information provided by a nominal model is inaccurate. It is ubiquitous for a nominal SDS to possess both inaccurate state evolution functions and incomplete distributional information of noises.For example,the nominal state evolution function may be a local linearization from a nonlinear model [8].Additionally, one's nominal information about the distributions of system noises is usually partial. Most of the time,only the estimated mean and covariance are available in the nominal model,which is far from uniquely determining a pdf [9]. A nominal predictor can lead to misleading predictive pdfs even though the uncertainties on SDS are not prominent (as demonstrated by experiments in Sec. VIII). Addressing this issue is crucial for the widespread application of probabilistic prediction in model-uncertain scenarios. In this paper,the aim is to develop a framework to design probabilistic predictors for SDSs with worst-case performance guarantees.

To design probabilistic predictors against distributional uncertainties of the model, we inherit the key notions from the distributionally robust optimization (DRO） [10]. DRO isa minimax-based mathematical framework for decision-making under uncertainty. It aims to find the decision that performs well across a range of possible pdfs in an ambiguity set. Nevertheless,what we need is to find the optimal probabilistic predictor that performs well over a set of SDSs. In DRPP, we generalize the idea of ambiguity sets to jointly describe one's uncertainties of noises’ pdfs and state evolution functions.

The proposed DRPP framework has direct applicability in control and decision-making systems.Reliable predictive pdfs and confidence regions produced by DRPP can be integrated into downstream modules such as risk-aware model predictive control,robot trajectory planning,and safety assessment in power-system frequency regulation.For instance,understanding the evolution of robust confidence regions enables safe motion planning for manipulators and informs constraint tightening in safety-critical control.

# C. Contributions

Motivated by the aforementioned considerations，we propose a novel functional-maximin-based distributionally robust probabilistic prediction (DRPP) framework.In DRPP, the objective is the expected score under the worst-case SDS within an ambiguity set, and the predictor optimizes the worst-case objective over the space of admissible predictive pdfs. The main contributions of this paper are summarized as follows:

· To the best of our knowledge,this is the first work on probabilistic prediction of SDSs that optimizes the worst-case performance over an ambiguity set. We have generalized the classical conic moment-based ambiguity set such that both the uncertainties of state evolution functions and system noises are quantified.   
We develop a methodology to greatly reduce the complexity of solving DRPP. First,we use the principle of dynamic programming to derive the Bellman equation. Second，we exploit a necessary optimality condition to equivalently transform the maximin problem from function spaces to Euclidean spaces.   
We design two suboptimal DRPP predictors by relaxing the constraints.Noise-DRPP is the optimal predictor when the nominal state evolution function is accurate, and Eig-DRPP is the optimal predictor when the predictor is constrained by an eigenvector restriction.Moreover, an upper bound of the optimality gap of the proposed predictors is obtained.

# D. Organization

The rest of this article is organized as follows.Section II provides a brief review of the related work and Section III introduces preliminaries on probabilistic prediction and DRO. In Section IV, we formulate the proposed DRPP problem.In Section V, we introduce the main methodology to address the main challenges in the DRPP.Then,we relax the constraints to solve two suboptimal DRPPs with upper and lower bounds in Section VI and Section VII. Next, we conduct elaborate experiments and provide an application in Section VIII, followed by a conclusion in Section IX.Please see Fig.1 for the overall structure of this paper.

![](images/4eb02e544a313a153917318f2397bf43dc82b53dd2830b6bceb3fe0ad358ec2f.jpg)  
Fig.1: An illustration of the roadmap.

# II. RELATED WORK

The study of DRPP for SDSs integrates ideas from fields including probabilistic prediction，control theory，minimax optimization, and DRO. In this section, we briefly summarize the most relevant work as follows.

a）Probabilistic prediction of SDS: According to how the predictive pdfs are approximated, existing literature in the control community can be classified into the following groups. First,approximation by the first several central moments. Particularly, there are many methods using the mean and covariance to describe the predictive distribution. Classical local approximation methods include the Taylor polynomial in the extended Kalman filter [11],[12],the Stirling's interpolation [13] and the unscented transformation [14] as representatives for derivative-free estimation methods [15].

Second,approximation by a weighted sum of basis functions.The Gaussian mixture model (GMM) [16] approximates a pdf by a sum of weighted Gaussian distributions,and it can achieve any desired approximation accuracy if there are sufficiently large number of Gaussians [17]. Using GMM, the probabilistic prediction is equivalent to propagating the first two moments and weights of each basis [18]. The polynomial chaos expansion (PCE) [19] approximates a state evolution function by a sum of weighted polynomial basis functions, and these basis functions are orthogonal regarding the input pdf.In this way, the mean and covariance of the output can be conveniently approximated by the weights of PCE [20].

Third,approximation by a finite number of sampling points. The numerical solution using Monte Carlo (MC) samplings [21] can approximate any statistics of the output distribution if the sampling number is suficiently high.For a further review of the modern MC methods in approximated probabilistic prediction,we recommend the review [22].

Nevertheless,assuming the accuracy of a nominal model is restrictive,especially in the control engineering practice. As reviewed in [7],an important application of probabilistic prediction for SDSs is to formulate chance constraints [23] in stochastic model predictive control (SMPC) [24], [25]. In these situations,a misleading predictive pdf can lead to unsafe control policies.To take the distributional uncertainties into account,Coulson et al [26] have proposed a novel distributionally robust data-enabled predictive control algorithm, Coppens and Patrinos [27] develop a data-driven distributionally robust MPC scheme using generalized moment-based ambiguity sets.However, these works aim to robustify the control performance，while the problem of distributionally robust probabilistic prediction remains open.

b） Probabilistic prediction in machine learning: Taking a historical view of the development of PP in the statistics and machine learning communities,we can find that the research in the early phase also focuses on approximating the posterior pdf, especially from a Bayesian perspective. The advent of the scoring rule has provided a convenient scalar way to measure the performance of probabilistic predictions [5].Then，the research focus is shifting toward training probabilistic predictors to optimize the empirical scores.Classical statistical methods include Bayesian statistical models [28],approximate

Bayesian forecasting [29],quantile regression [3O], etc.Recent machine learning algorithms include distributional regression [31], boosting methods such as NGBoost [32],and autoregressive recurrent networks like DeepAR[33], etc.We recommend [34] for a more detailed review of machine learning algorithms for probabilistic prediction.

c）Distributionally robust optimization: To guarantee the prediction performance for SDSs when the nominal model is not accurate, the primary task is to describe the ambiguity set for SDSs,which is the key concept in the DRO community. In the seminal work [35], the Chebyshev ambiguity set is defined, where only the mean and covariance are known. Using this ambiguity set, several extensions are further studied in [36]. Next,Delage and Ye [1O] generalize the Chebyshev ambiguity set to the conic moment-based ambiguity set, which allows the mean and covariance matrix to be also unknown. Shapiro and Pichler [37] further extend the conic moment-based ambiguity set to the conditional version,which is suitable for multistage decision-making problems. Overall,viewing from the outer optimization, DRO still optimizes in an Euclidean space,while DRPP optimizes in a functional space of all pdfs. Since classical tools in DRO cannot be directly applied to solve DRPP, new methods are needed.

holds for all $\hat { p } _ { { \pmb x } } , p _ { { \pmb x } } \in { \mathcal F }$ .It is strictly proper if the equality holds only when $\hat { p } _ { x }$ equals $p _ { x }$ almost everywhere.For example,the logarithm score,

$$
\begin{array} { r } { \mathcal { L } ( \hat { p } _ { \pmb { x } } , x ) : = \log \hat { p } _ { \pmb { x } } ( x ) , } \end{array}
$$

is one of the most celebrated scoring rules for being essentially the only local proper scoring rule up to equivalence [38].

# B. Distributionally Robust Optimization

Traditional stochastic optimization approaches assume a known probability distribution $P$ and solve problems

$$
\operatorname* { i n f } _ { x \in \mathcal { X } } \mathbb { E } _ { \xi \sim P } [ h ( x , \xi ) ] ,
$$

where $x \in \mathcal { X }$ represents the decision variable, $\xi$ is sampled from the distribution $P$ ， $h ( x , \xi )$ is the objective function. However,in many real-world problems, the true probability distribution of the uncertain parameters is unknown or partially known. DRO addresses this issue by considering a family of distributions,known as an ambiguity set,rather than a single distribution. The goal is to find a solution that performs well for the worst-case distribution within this set.A general DRO problem is formulated as:

# III. PRELIMINARIES

In this paper, we denote random variables in bold fonts to distinguish them from constant variables.Given a random variable $_ { x }$ taking values in $\mathcal { X }$ ，we denote its probability measure as $P _ { \mathbf { \delta x } } ( \cdot )$ and its pdf as $p _ { \mathbf { x } } ( \cdot )$ . Condsider another random variable $\textbf {  { y } }$ ,we deonte the conditional probability measure of $\textbf {  { y } }$ given $_ { x }$ as $P _ { \mathbf { x } | \mathbf { y } } ( \cdot \mid \cdot )$ and the conditional pdf as $p _ { \pmb { x } | \pmb { y } } ( \cdot | \cdot )$ Let $\mathcal { P } ( \mathcal { X } )$ be the space of probability measures on $\mathcal { X }$ ， equipped with the standard weak\*-topology.Let $\mathcal { P } _ { 2 } ( \mathcal { X } ) \subset \mathcal { P } ( \mathcal { X } )$ be the space of probability measures with finite second moments. Let $\mathcal { M } _ { + } ( \mathcal { X } )$ be the set of positive measures on $\mathcal { X }$ . Given a measurable map $h : \mathcal { X }  \mathbb { R }$ ，the expectation of $h ( { \pmb x } )$ is denoted as $\mathbb { E } _ { P _ { \mathbf { x } } } h ( { \boldsymbol { x } } )$ or $\mathbb { E } _ { x \sim p _ { x } } h ( x )$ . We also denote a sequence $\{ ( \cdot ) _ { k } \} _ { k = 1 } ^ { T }$ by $( \cdot ) _ { 1 : T }$ . We denote the set of symmetric matrices with dimension $n$ as $S ^ { n }$ , whose subset of positive semi-definite matrices is $S _ { + } ^ { n }$ . For a function $f : \mathcal { X } \ \to \ \mathbb { R }$ ，the operator norm $\| f \| _ { 2 } : = \operatorname* { i n f } \{ c \geq 0 : \| f ( v ) \| _ { 2 } \leq c \| v \| _ { 2 } \forall v \in \mathbb { R } ^ { n } \} .$ For $\rho \in \mathcal { P } ( \mathcal { X } )$ ，we denote $\langle \rho , f \rangle : = \int _ { \mathcal { X } } f ( x ) \mathrm { d } \rho ( x )$ . Given $A , B \in \mathbb { R } ^ { n \times n }$ ,we use $A \succeq B$ to indicate $A - B \in S _ { + } ^ { n }$ ，and we denote $\langle A , B \rangle : = \operatorname { t r } ( A ^ { \top } B )$

# A．Probabilistic Prediction

Given a random variable $_ { x }$ taking value on $\mathcal { X }$ ，consider a predictive space, $\mathcal { F }$ ，of probability measures on $\mathcal { X }$ that admits a Lebesgue density, $\hat { p } _ { x }$ ,for each element. After the value of $_ { \pmb { x } }$ is materialized as $x$ , a scoring rule,

$$
S ( \hat { p } _ { \mathbf x } , x ) : \mathcal F \times \mathcal X \to \mathbb R ,
$$

assigns a numerical score $S ( \hat { p } _ { x } , x )$ to measure the quality of the predictive distribution $\hat { p } _ { { \pmb x } }$ on the realized value $x$ . Notice that the scoring rule $S ( \hat { p } _ { x } , x )$ is a random variable because it depends on the realization of $_ { x }$ . A scoring rule $s$ is proper with respect to the predictive space $\mathcal { F }$ if $\mathbb { E } S ( \hat { p } _ { \pmb { x } } , \pmb { x } ) \le \mathbb { E } S ( p _ { \pmb { x } } , \pmb { x } )$

$$
\operatorname* { i n f } _ { x \in \mathcal { X } } \operatorname* { s u p } _ { Q \in \mathcal { D } } \mathbb { E } _ { \xi \sim Q } [ h ( x , \xi ) ] ,
$$

where $\mathcal { D }$ is the ambiguity set, a set of probability distributions that are close to the nominal distribution $P$ in some sense.

The ambiguity set $\mathcal { D }$ can be defined in various ways. One of the most popular ones is the moment-based sets,where distributions are constrained by their moments (e.g.，mean, variance).For example, the celebrated conic moment-based ambiguity set is used in the seminal work [1O] such that

$$
\begin{array} { r l } & { \mathcal { D } _ { 1 } \left( \mathcal { X } , \mu _ { 0 } , \Sigma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } \right) } \\ & { : = \left\{ \begin{array} { l l } { \begin{array} { r l } { p _ { \pm } \left( \pm \xi \in \mathcal { X } \right) = 1 } & { } \\ { p _ { \pm } \left( \mathbb { E } \left[ \xi \right] - \mu _ { 0 } \right) ^ { \top } \Sigma _ { 0 } ^ { - 1 } \left( \mathbb { E } \left[ \xi \right] - \mu _ { 0 } \right) \leq \gamma _ { 1 } } \\ { \mathbb { E } \left[ \left( \xi - \mu _ { 0 } \right) \left( \xi - \mu _ { 0 } \right) ^ { \top } \right] \preceq \gamma _ { 2 } \Sigma _ { 0 } } & \end{array} } \end{array} \right\} , } \end{array}
$$

where $\mathcal { X }$ is the nominal support, $\mu _ { 0 }$ and $\Sigma _ { 0 }$ are the nominal mean and covariance, $\gamma _ { 1 } , \gamma _ { 2 } \in \mathbb { R } _ { + }$ quantifies the uncertainties on the mean and covariance.A thorough review of popular ambiguity sets is provided in [39, Sec. 3].

# IV.PROBLEM FORMULATION

To begin with，we define the system and probabilistic predictor model, respectively. Next, we generalize the classical conic moment-based ambiguity set (1） for SDSs such that one's uncertainties of both state evolution functions and system noises are jointly described.Finally,we formulate the DRPP problem as a multistep functional maximin optimization.

# A. System and Predictor Model

System model $( \mathcal { X } , \mathcal { U } , \mathcal { P } , \pi , T )$ .Consider a discrete-time SDS with horizon $T \in \mathbb { Z } _ { + }$ ， denoted as $\mathcal { P }$ ，whose dynamics (or kernel) at time step $k \in \{ 0 , \ldots , T - 1 \}$ is

$$
\begin{array} { r } { \mathscr { P } _ { k } : \pmb { x } _ { k + 1 } = f _ { k } ( \pmb { x } _ { k } , \pmb { u } _ { k } ) + \pmb { w } _ { k } , } \end{array}
$$

where $\mathcal { P } _ { k }$ denotes the system kernel at time step $k$ ， $\scriptstyle { \mathbf { { \mathit { x } } } } _ { k }$ is the system state taking values in the state space $\mathcal { X } : = \mathbb { R } ^ { d _ { x } }$ ，the initial state is given as $x _ { 0 } , \pmb { u } _ { k }$ is the control taking values in the input space $\mathcal { U } : = \mathbb { R } ^ { d _ { u } }$ ，which comes from a deterministic state-feedback policy $\pi _ { k } : \mathcal X \to \mathcal U$ ，i.e.,， ${ \pmb u } _ { k } \ = \ \pi _ { k } ( { \pmb x } _ { k } )$ The subsequent problem formulation thus evaluates predictive performance conditioned on a fixed policy $\pi$ .We denote the state-control pair $( \boldsymbol { x } _ { k } , \boldsymbol { u } _ { k } )$ as $z _ { k } \ \in \ \mathcal { Z } \ : = \ \mathcal { X } \times \mathcal { U }$ ，then $f _ { k } : { \mathcal { Z } } \to { \mathcal { X } }$ is the state evolution function. ${ \pmb w } _ { k }$ isan exogenous noise vector taking values in $\mathbb { R } ^ { d _ { x } }$

Assumption 1 (SDS). At each time step $k \in \{ 0 , . . . , T - 1 \}$ · $f _ { k }$ has finite operator norms,i.e., $\| f _ { k } \| _ { 2 }$ is bounded, which is satisfied by many nonlinear systems in practice. . the noise ${ \pmb w } _ { k }$ has a bounded second moment, i.e., $P _ { w _ { k } } \in$ $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d _ { x } } )$ ，and ${ \pmb w } _ { 0 : T - 1 }$ are independent but not necessarilyidentically distributed.

Predictor model $\textstyle ( { \mathcal { F } } )$ . Starting from the system's initial state $x _ { 0 }$ ，a probabilistic predictor keeps observing the states and the control inputs,aiming to predict the conditional pdfs of the next state.

Definition 1 (Predictive space). Let $\mathcal { F } \subset \mathcal { P } ( \mathcal { X } )$ be the set of continuous positive pdfs on $\mathcal { X }$ with tail envelope of order 2, i.e.,

$$
\mathcal { F } : = \left. \hat { p } \ \middle | \begin{array} { l } { \hat { p } \ i s \ a \ s t r i c t l y \ p o s i t i \nu e \ p d f o n \ \chi , } \\ { \exists C > 0 \ s . t . | \log \hat { p } ( x ) | \leq C ( 1 + \| x \| _ { 2 } ^ { 2 } ) \ \forall x \in \chi } \end{array} \right. \ ,
$$

A probabilistic predictor policy for $\mathcal { P }$ is a sequence $\mathcal { F } =$ $\left( \mathcal { F } _ { 0 } , \ldots , \mathcal { F } _ { T - 1 } \right)$ where each $\mathcal { F } _ { k } : \mathcal { Z }  \mathcal { F }$ is a measurable map from the previous state and control input to a predictive conditional pdf for the next state.Let $\mathfrak { F }$ be the collection of such predictor policies.

Assumption 2 (Predictor's prior information). At each time step $k \in \{ 0 , \ldots , T - 1 \}$ ，the probabilistic predictor has only access to the following information:

· the state trajectory $x _ { 0 : k }$ ， control input sequence $u _ { 0 : k }$ ， · a nominal state evolution function ${ \overline { { f } } } _ { k } : { \mathcal { Z } } \to { \mathcal { X } }$ ， · a nominal noise mean $\bar { \mu } _ { k } \ \in \ \mathcal { U }$ ，and a nominal noise covariance $\bar { \Sigma } _ { k } \in S _ { + } ^ { d _ { x } }$

The predictor's prior information usually does not align with the ground truth. Therefore, to further quantify the uncertainty between the nominal model and the real model,we need to define an ambiguity set of SDSs.

# B. Ambiguity Set

The crucial spirit of an ambiguity set is that although a nominal model rarely identifies with the ground truth,one can still have additional information or belief that the real model is located around the nominal model. For example,an upper bound for the operator norm $\| f _ { k } - \bar { f } _ { k } \| _ { 2 }$ may be available when the nominal function is derived from a system identification procedure.Additionally，estimations and confidence regions (CRs）of the mean and covariance of system noise may be statistically inferred from historical data. Even if no supportive evidence exists to construct a reasonable ambiguity set, one can still subjectively select an ambiguity set.

Jointly considering the uncertainties of both state evolution functions and noises,we define the conditional conic momentbased ambiguity sets for $\mathcal { P }$

Definition 2 (Conditional conic moment-based ambiguity set). For each time step $k \in \{ 0 , \ldots , T - 1 \}$ and state-control pair $z$ ， the conditional conic moment-based ambiguity set is a subset of $\mathcal { P } _ { 2 } ( \mathcal { X } )$ defined as

$$
\left\{ \begin{array} { l } { \displaystyle \mathcal { T } _ { k } \left( z \mid \bar { f } _ { k } , \bar { \mu } _ { k } , \bar { \Sigma } _ { k } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } , \gamma _ { 3 } \right) = : } \\ { \displaystyle \int _ { P _ { \mathbf { x } _ { k + 1 } | z _ { k } } \left( \cdot \vert z \right) } \left. \begin{array} { l } { \displaystyle x _ { k + 1 } = f _ { k } ( z ) + w _ { k } , } \\ { \displaystyle \left. f _ { k } ( z ) - \bar { f } _ { k } ( z ) \right. _ { 2 } ^ { 2 } \leq \gamma _ { 0 } ( z ) } \\ { \displaystyle \left. \mathbb { E } ( w _ { k } ) - \bar { \mu } _ { k } \right. _ { \bar { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \leq \gamma _ { 1 } } \\ { \displaystyle \gamma _ { 3 } \bar { \Sigma } _ { k } \preceq \mathbb { E } \left( w _ { k } - \bar { \mu } _ { k } \right) \left( w _ { k } - \bar { \mu } _ { k } \right) ^ { \top } \preceq \gamma _ { 2 } \bar { \Sigma } _ { k } } \end{array} \right\} , } \end{array} \right.
$$

where $\bar { f } _ { k } , \bar { \mu } _ { k } , \bar { \Sigma } _ { k }$ are defined in Assumption 2,and predictor's uncertainties are quantified by $\gamma _ { 0 } : \mathcal { Z } \to \mathbb { R } _ { + }$ and $\gamma _ { j } \in \mathbb { R } _ { + }$ for $j = 1 , 2 , 3$ ，When there is no confusion about the parameters that define an ambiguity set, we use $\mathcal { T } _ { k } ( z )$ for simplification.

The conic moment-based ambiguity set covers a wide family of distributions with bounded second moments.This choice not only facilitates tractable dual reformulations but also provides a statistically interpretable description of moment uncertainty, leading to guaranteed robustness without assuming a specific parametric form.

Assumption 3 (Regularity). At each time step $k \in \{ 0 , \ldots , T -$ $1 \}$ ，conditioned on the state-control pair $z _ { k } = z _ { k }$ ，the system chooses a conditional probability measure $P _ { k } ( \cdot \mid z _ { k } ) \in \mathcal { T } _ { k } ( z _ { k } )$ indepentdent of the previous choices.

Let $\mathfrak { P }$ be a collection of SDSs subject to the conditional conic moment-based ambiguity set and regularity assumption.

# C.Problem in Interest

Objective function $( J _ { k } ^ { \mathcal { F } , \mathcal { P } } )$ Starting from the time step $k \in \{ 0 , \ldots , T - 1 \}$ ， given an initial state-control pair $z \in$ $\mathcal { Z }$ ，apredictor policy $\mathcal { F } \in \mathfrak { F }$ ，and a system kernel $\mathcal { P } \in$ $\mathfrak { P }$ ，the prediction performance of $\mathcal { F }$ is characterized by the expectation of cumulative log-score over the state trajectory. We define it by the objective function as follows:

$$
J _ { k } ^ { \mathcal { F } , \mathcal { P } } ( z ) : = \mathbb { E } _ { \mathcal { P } _ { k : T - 1 } } \left[ \sum _ { t = k } ^ { T - 1 } \mathcal { L } \left( \mathcal { F } _ { t } ( z _ { t } ) , \pmb { x } _ { t + 1 } \right) \bigg | z _ { k } = z \right] ,
$$

where the notation $\mathbb { E } _ { \mathcal { P } }$ indicates that the expectation is taken over the random trajectory $( \pmb { x } _ { k } , \dots , \pmb { x } _ { T } )$ generated by the kernel sequence $\mathcal { P } _ { k : T - 1 }$

Distributionally robust probabilistic prediction $\left( \mathbf { P } _ { 0 } \right)$ Given a set of probabilistic predictors $\mathfrak { F }$ and a set of system kernels $\mathfrak { P }$ for the SDS $\mathcal { P }$ with an initial state-control pair $z _ { 0 }$ ， we are interested in finding probabilistic predictors in $\mathfrak { F }$ that are distributionally robust with respect to $\mathfrak { P }$ .In other words, a distributionally robust probabilistic predictor should have a worst-case performance guarantee for any system kernel in $\mathfrak { P }$ . To this end,a multistep functional maximin optimization is formulated as follows:

$$
( \mathbf { P } _ { 0 } ) : \operatorname* { s u p } _ { \mathcal { F } \in \mathfrak { F } } \operatorname* { i n f } _ { \mathcal { P } \in \mathfrak { P } } J _ { 0 } ^ { \mathcal { F } , \mathcal { P } } ( z _ { 0 } ) .
$$

# V. MAIN METHODOLOGY

To begin with,we derive the Bellman equation of DRPP based on the principle of dynamic programming,which shows that solving the DRPP problem $\mathbf { P } _ { 0 }$ is equivalent to solving a sequence of Bellman equations in a backward manner. Then, we canonicalize the ambiguity set of SDS and transform the inner optimization into an infinite-dimensional conic linear program. Next,we take the dual form of inner minimization with a strong duality guarantee.A necessary optimality condition is developed to handle the computational intractability of functional optimization. Exploiting this optimality condition, we equivalently transform the one-step DRPP in function spaces into a convex-nonconcave minimax problem in Euclidean spaces.Finally，a discussion on the computational complexity is provided.

# A. Bellman Equation

The concept of robust value function originates from the study of robust Markov decision processes (RMDPs)，where the optimal worst-case performance of a control problem is quantified. Here, we apply the same idea to $\mathbf { P } _ { 0 }$ . Starting from the time step $k \in \{ 0 , \ldots , T - 1 \}$ ， given an initial state-control pair $z \in { \mathcal { Z } }$ and a predictor policy $\mathcal { F } \in \mathfrak { F }$ ，the worst-case objective function for $\mathcal { F }$ is defined as

$$
V _ { k } ^ { \mathcal { F } } ( z ) : = \operatorname* { i n f } _ { \mathcal { P } \in \mathfrak { P } } J _ { k } ^ { \mathcal { F } , \mathcal { P } } ( z ) .
$$

Then,the robust value function is defined as the optimal worstcase objective function, i.e.,

$$
V _ { k } ^ { \ast } ( z ) : = \operatorname* { s u p } _ { \mathcal { F } \in \mathfrak { F } } V _ { k } ^ { \mathcal { F } } ( z ) .
$$

Leveraging the principle of dynamic programming, one can derive the following recursive equations for $V _ { k } ^ { * }$

Theorem 1 (Bellman equation). The set of robust value functions $\{ V _ { k } ^ { * } , k = 0 , \ldots , T \}$ satisfies the following Bellman equation: for any state-control pair $z \in { \mathcal { Z } }$ ， $V _ { T } ^ { * } ( z ) = 0 ,$ and for $k \in \{ 0 , \ldots , T - 1 \}$ ，thereis

$$
V _ { k } ^ { * } ( z ) = \operatorname* { s u p } _ { \hat { p } _ { k } \in \mathcal { F } } \operatorname* { i n f } _ { \rho _ { k } \in \mathcal { Z } _ { k } ( z ) } \int _ { \mathcal { X } } \mathcal { L } ( \hat { p } _ { k } , x ) + V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \mathrm { d } \rho _ { k } ( x ) .
$$

Proof. Please see Appendix A.

Remark 1. The Bellman equation implies that diffrent control policies can lead to different robust value functions even when the system kernel family $\mathfrak { P }$ and predictor family $\mathfrak { F }$ are fixed.Thus,theDRPP $\mathbf { P } _ { 0 }$ can be extended to jointly optimize the control policy and the predictor policy in future work.

Theorem 1 shows that solving the DRPP problem $\mathbf { P } _ { 0 }$ is equivalent to solving the Bellman equation (7). This result generalizes the classical Bellman equation for RMDPs [40] to the DRPP setting.

Let $\nu _ { k } = f _ { k } ( z )$ and $\begin{array} { r } { \overline { { \nu } } _ { k } = \overline { { f } } _ { k } ( z ) } \end{array}$ , the ambiguity constraint of $f _ { k }$ is equivalent to describing the distance between $\nu _ { k }$ and $\bar { \nu } _ { k }$ ， i.e., $\| \nu _ { k } - \bar { \nu } _ { k } \| _ { 2 } ^ { 2 } = \| f _ { k } ( z ) - \bar { \bar { f } } _ { k } ( z ) \| _ { 2 } ^ { 2 } \leq \gamma _ { 0 } ( z )$ . To deal with the remaining two ambiguity constraints on the first two moments of ${ \pmb w } _ { k }$ ,let $\mu _ { k } = \mathbb { E } w _ { k }$ ,and the r.h.s. of(7) is transformed as

$$
\begin{array} { r l } & { \underset { \bar { \mathcal { P } } k \in \mathcal { F } ^ { \nu _ { k } , \mu _ { k } , \rho _ { k } } } { \operatorname* { s u p } } \underset { \mathcal { P } k } { \operatorname* { i n f } } \ \underset { ( \mathcal { X } ( \hat { p } _ { k } , x ) \underset { \mathcal { P } _ { k } } { \operatorname* { i n f } } [ 1 ] = 1 , \mathbb { E } _ { x \times \rho _ { k } } [ x ] = \mu _ { k } + \nu _ { k } } } \\ & { \underset { \mathrm { s . t . } } { \operatorname* { s u p } } \underset { \mathcal { P } _ { 3 } \underset { \mathbf { \bar { Z } } _ { k } \leq \mathcal { B } _ { k } \leq \mathcal { P } _ { k } } { \operatorname* { i n } } } \left[ ( x - \nu _ { k } - \bar { \mu } _ { k } ) ( x - \nu _ { k } - \bar { \mu } _ { k } ) ^ { \top } \right] \preceq \gamma _ { 2 } \bar { \Sigma } _ { k } } \\ &  \mathrm { s . t . } \left\{ \begin{array} { l l } { \begin{array} { r l } { \bar { \Sigma } _ { k } } & { ( \mu _ { k } - \bar { \mu } _ { k } ) } \\ { \left( \mu _ { k } - \bar { \mu } _ { k } \right) ^ { \top } } & { \gamma _ { 1 } } \\ { \left[ \begin{array} { l l } { I } & { ( \nu _ { k } - \bar { \nu } _ { k } ) } \\ { ( \nu _ { k } - \bar { \nu } _ { k } ) ^ { \top } } & { \gamma _ { 0 } ( z ) } \end{array} \right] \succeq 0 , } \end{array} \right. } \end{array} \end{array}
$$

where the inner problem is canonicalized into an infinitedimensional conic linear program.

# B. Optimality Condition

Directly solving the worst-case probability measure in the positive measure space $\mathcal { M } _ { + } ( \mathcal { X } )$ is computationally intractable. As a first step towards dealing with this challenge, we leverage the dual analysis for (8). If a strong duality holds,one can optimize the Lagrange multipliers in the dual form.For the inner optimization of (8),a classical conclusion is that $\bar { \Sigma } _ { k } \succ 0$ is a sufficient condition for strong duality to hold [1O, p.5].Let $r \in \mathbb { R } , q \in \mathbb { R } ^ { d _ { x } } , Q _ { i } \in S _ { + } ^ { d _ { x } } , \kappa _ { i } = \{ P _ { i } \stackrel { \cdot } { \in } S _ { + } ^ { d _ { x } } , p _ { i } \in \hat { \mathbb { R } } ^ { d _ { x } } , s _ { i } \in \hat { \mathbb { R } } ^ { d _ { x } } ,$ $\mathbb { R } \}$ for $i = 1 , 2$ be the Lagrange multipliers,the dual form of (8)is

$$
\begin{array} { r l } & { ( \mathbf { D } _ { 1 } ) : \underset { \hat { p } _ { k } \in \mathcal { F } , r , q , Q _ { 1 } , Q _ { 2 } , \kappa _ { 1 } , \kappa _ { 2 } } { \operatorname* { s u p } } G ( r , q , Q _ { 1 } , Q _ { 2 } , \kappa _ { 1 } , \kappa _ { 2 } ) } \\ & { \qquad \mathrm { s . t . } \left\{ \begin{array} { l l } { x ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) x + x ^ { \top } q + r + \log \hat { p } _ { k } ( x ) } \\ { + V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \geq 0 \forall x \in \mathcal { X } } \\ { \left[ \begin{array} { l l } { P _ { 1 } } & { p _ { 1 } } \\ { p _ { 1 } ^ { \top } } & { s _ { 1 } } \end{array} \right] \succeq 0 , \left[ \begin{array} { l l } { P _ { 2 } } & { p _ { 2 } } \\ { p _ { 2 } ^ { \top } } & { s _ { 2 } } \end{array} \right] \succeq 0 . } \end{array} \right. } \end{array}
$$

wher $\ L \circ G ( r , q , Q _ { 1 } , Q _ { 2 } , \kappa _ { 1 } , \kappa _ { 2 } ) = \ L - r + \bar { \mu } _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \bar { \mu } _ { k } - ( \gamma _ { 2 } Q _ { 1 } -$ $\gamma _ { 3 } Q _ { 2 } + P _ { 1 } ) \cdot \bar { \Sigma } _ { k } - P _ { 2 } \cdot I + 2 p _ { 1 } ^ { \top } \bar { \mu } _ { k } - s _ { 1 } \gamma _ { 1 } + 2 p _ { 2 } ^ { \top } \bar { \nu } _ { k } - s _ { 2 } \gamma _ { 0 } ( z ) ^ { 2 } +$ $g ( q , p _ { 1 } , p _ { 2 } , Q _ { 1 } , Q _ { 2 } )$ ，and $g ( q , p _ { 1 } , p _ { 2 } , Q _ { 1 } , Q _ { 2 } ) = \mathrm { i n f } _ { \nu _ { k } } - ( q +$ $2 p _ { 2 } ) ^ { \top } \nu _ { k } - \nu _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \nu _ { k }$ s.t. $q + 2 p _ { 1 } + 2 ( Q _ { 1 } - Q _ { 2 } ) ( \nu _ { k } + \bar { \mu } _ { k } ) =$ O.Please see Appendix B for a detailed derivation.

Unfortunately,we have to deal with an infinite number of constraints indexed by $x \in \mathcal { X }$ in problem $\mathbf { D } _ { 1 }$ . Although the dual problem cannot lead to a direct solution, it helps to develop an optimality condition for the probabilistic predictors in the following theorem.

Theorem 2 (Optimality condition). $A$ necessary optimality condition for the DRPP $\mathbf { P } _ { 0 }$ is that $\hat { p } _ { k }$ is belongs to the following exponential family almost everywhere on $\mathbb { R } ^ { d _ { x } }$ ,

$$
\begin{array} { r } { \hat { p } _ { k } ( x ) \stackrel { a . s . } { \propto } \exp \{ - x ^ { \top } \theta _ { 1 } x - x ^ { \top } \theta _ { 2 } - V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \} , } \end{array}
$$

where $\propto$ means the right-hand side gives the density up to a normalizing constant, $\theta _ { 1 } ~ \in ~ S ^ { d _ { x } } , ~ \theta _ { 2 } ~ \in ~ \mathbb { R } ^ { d _ { x } }$ ，and $k \in \{ 0 , \ldots , T - 1 \}$ . Particularly for $k = T - 1$ ， the optimial $\hat { p } _ { T - 1 }$ is subject to a Gaussian distribution almost everywhere, i.e., $\exists \hat { \mu } _ { k } \in \mathbb { R } ^ { d _ { x } }$ and $\hat { \Sigma } _ { k } \in S _ { + } ^ { d _ { x } }$ such that

$$
\begin{array} { r } { \hat { p } _ { k } \overset { a . s . } { \sim } \mathcal N \left( \hat { \mu } _ { k } , \hat { \Sigma } _ { k } \right) . } \end{array}
$$

Proof. Please see Appendix C.

Theorem 2 has reduced the optimization complexity from the original function space $\mathcal { F }$ to a specific exponential distribution family. For $k = T - 1$ ，the space is further reduced to $S _ { + } ^ { d _ { x } } \times \mathbb { R } ^ { \bar { d } _ { x } }$ ，where $\mathbf { D } _ { 1 }$ can be transformed into a finitedimensional optimization problem as follows

$$
\begin{array} { l } { { \displaystyle \operatorname* { m a x } _ { \hat { \mu } _ { k } , \hat { \Sigma } _ { k } , Q _ { 1 } , Q _ { 2 } , \kappa _ { 1 } , \kappa _ { 2 } } - \frac { 1 } { 2 } d _ { x } \log ( 2 \pi ) + \frac { 1 } { 2 } \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ^ { - 1 } ) } } \\ { { \displaystyle - \left( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } \right) \cdot \bar { \Sigma } _ { k } - P _ { 1 } \cdot \bar { \Sigma } _ { k } - P _ { 2 } \cdot I - s _ { 1 } \gamma _ { 1 } } } \\ { { \displaystyle - s _ { 2 } \gamma _ { 0 } ( z ) ^ { 2 } + 2 p _ { 1 } ^ { \top } \hat { \Sigma } _ { k } ( 2 p _ { 2 } - p _ { 1 } ) - 2 p _ { 2 } ^ { \top } ( \hat { \mu } _ { k } - \bar { \mu } _ { k } - \bar { \nu } _ { k } ) } } \\ { { \displaystyle \mathrm { s . t . } \left\{ Q _ { 1 } \beth _ { 2 } \ b Q _ { 2 } \ b \geq 0 , Q _ { 1 } - Q _ { 2 } = \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } , \hat { \Sigma } _ { k } \succ 0 \right. } } \\ { { \displaystyle \left[ \begin{array} { c c } { { P _ { 1 } } } & { { p _ { 1 } } } \\ { { p _ { 1 } ^ { \top } } } & { { s _ { 1 } } } \end{array} \right] \succeq 0 , \left[ \begin{array} { c c } { { P _ { 2 } } } & { { p _ { 2 } } } \\ { { p _ { 2 } ^ { \top } } } & { { s _ { 2 } } } \end{array} \right] \succeq 0 . } } \end{array}
$$

Please see Appendix $\mathrm { D }$ for a detailed derivation. However, this problem is now a nonconvex semidefinite optimization with respect to $\hat { \Sigma } _ { k }$ , solving a globally optimal solution is NPhard in general. Even worse, the lack of a closed-form solution of $V _ { T - 1 } ^ { * }$ makes it impossible to recursively parameterize and solve $\hat { p } _ { k }$ backwardly based on the Bellman equation (7).

# C. One-step DRPP

At time step $k \in \{ 0 , \ldots , T - 1 \}$ , the Bellman equation （7) implies that the optimal prediction policy maximizes the onestep score $\mathcal { L } ( \hat { p } _ { k } , x )$ plus the expected best future cumulative scores $V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) )$ from the next state $x$ onward. If one does not take into account the future scores, Theorem 2 shows that the optimal predictive pdf can be parametrized by a Gaussian distribution. Given $z \in { \mathcal { Z } }$ ，consider the following one-step DRPP problem:

$$
( \mathbf { P } _ { 1 } ) : \operatorname* { s u p } _ { \hat { p } _ { k } \in \mathcal { F } } \operatorname* { i n f } _ { \rho _ { k } \in \mathbb { Z } _ { k } ( z ) } \int _ { \mathcal { X } } \mathcal { L } ( \hat { p } _ { k } , x ) \mathrm { d } \rho _ { k } ( x ) ,
$$

we substitute $\hat { p } _ { k }$ by (10)and the objective follows as

$$
\mathbb { E } _ { { x } \sim { \rho } _ { k } } - \frac { 1 } { 2 } [ d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) + ( { x } - \hat { \mu } _ { k } ) \hat { \Sigma } _ { k } ^ { - 1 } ( { x } - \hat { \mu } _ { k } ) ^ { \top } ] .
$$

Next, notice that $\mathbb { E } _ { x \sim \rho _ { k } } [ x ] = \mu _ { k } + \nu _ { k }$ and $\mathbb { E } _ { x \sim \rho _ { k } } ( x - \nu _ { k } -$ $\mu _ { k } ) ( x - \nu _ { k } - \mu _ { k } ) ^ { \top } = \Sigma _ { k }$ one has $\mathbb { E } _ { x \sim \rho _ { k } } ( x - \hat { \mu } _ { k } ) \hat { \Sigma } _ { k } ^ { - 1 } ( x -$ $\hat { \mu } _ { k } ) ^ { \top }$ unealsto $[ \Sigma _ { k } + ( \mu _ { k } + \nu _ { k } - \hat { \mu } _ { k } ) ( \mu _ { k } + \nu _ { k } - \hat { \mu } _ { k } ) ^ { \top } ] \cdot \hat { \Sigma } _ { k } ^ { - 1 }$ ${ \bf P } _ { 1 }$ optimization in Euclidean spaces as follows

$$
\begin{array} { r l } & { ( { \mathbf { P } } _ { 2 } ) : \underset { \hat { { \Sigma } } _ { k } , \hat { \mu } _ { k } } { \operatorname* { m i n } } \ \underset { { \Sigma } _ { k } , \mu _ { k } , \nu _ { k } } { \operatorname* { m a x } } - \log \operatorname* { d e t } ( \hat { { \Sigma } } _ { k } ^ { - 1 } ) } \\ & { \qquad + \left[ { \Sigma } _ { k } + ( \mu _ { k } + \nu _ { k } - \hat { \mu } _ { k } ) ( \mu _ { k } + \nu _ { k } - \hat { \mu } _ { k } ) ^ { \top } \right] \cdot \hat { { \Sigma } } _ { k } ^ { - 1 } } \\ & { \qquad \mathrm { s . t . } \left\{ \begin{array} { l l } { \lVert \nu _ { k } - \bar { \nu } _ { k } \rVert _ { 2 } ^ { 2 } \leq \gamma _ { 0 } ( z ) , \lVert \mu _ { k } - \bar { \mu } _ { k } \rVert _ { \bar { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \leq \gamma _ { 1 } } \\ { \gamma _ { 3 } \bar { \Sigma } _ { k } \preceq { \Sigma } _ { k } + ( \mu _ { k } - \bar { \mu } _ { k } ) ( \mu _ { k } - \bar { \mu } _ { k } ) ^ { \top } \preceq \gamma _ { 2 } \bar { \Sigma } _ { k } . } \end{array} \right. } \end{array}
$$

Lemma 1. Given any $\mu _ { k }$ , the one-step DRPP $\mathbf { P } _ { 2 }$ has explicit solutions for $\Sigma _ { k }$ and $\hat { \mu } _ { k }$ as

Proof. Please see Appendix E.

Substituting the expressions of $\Sigma _ { k } ^ { * }$ and $\hat { \mu } _ { k } ^ { * }$ into $\mathbf { P } _ { 2 }$ ,it is equivalent to a convex-nonconcave minimax optimization:

$$
\begin{array} { r l } & { \underset { \hat { \Sigma } _ { k } \succ 0 } { \mathrm { m i n ~ m a x ~ } } - \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ^ { - 1 } ) + \gamma _ { 2 } \bar { \Sigma } _ { k } \cdot \hat { \Sigma } _ { k } ^ { - 1 } } \\ & { ~ + \left\| a + b \right\| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \left\| a \right\| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } } \\ & { \mathrm { s . t . } \ \| a \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \leq \gamma _ { 1 } , \| b \| _ { 2 } ^ { 2 } \leq \gamma _ { 0 } ( z ) , } \end{array}
$$

where $a ~ = ~ \mu _ { k } ~ - ~ \bar { \mu } _ { k } , b ~ = ~ \nu _ { k } ~ - ~ \bar { \nu } _ { k }$ . Notice that the inner maximization is an indefinite quadratic constrained quadratic programming (QCQP） problem，and there are still short of efficient algorithms to solve this kind of minimax optimization.

Given any $\hat { \Sigma } _ { k } \ \succ \ 0$ and $b \neq 0$ ，one can first solve $a$ by considering the following quadratic constrained linear programming (QCLP） problem:

$$
\operatorname* { m a x } _ { a } \| a + b \| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \| a \| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \mathrm { ~ s . t . ~ } \| a \| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \leq \gamma _ { 1 } .
$$

Lemma 2. The solution of the QCLP problem (13) is

$$
a ^ { * } = \frac { \sqrt { \gamma _ { 1 } } \bar { \Sigma } _ { k } \hat { \Sigma } _ { k } ^ { - 1 } b } { \Vert \bar { \Sigma } _ { k } \hat { \Sigma } _ { k } ^ { - 1 } b \Vert _ { \bar { \Sigma } _ { k } ^ { - 1 } } } .
$$

Proof. Using the KKT condition, there are

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \partial _ { a } \left[ \| a + b \| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \| a \| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - s _ { 1 } ( \| a \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \gamma _ { 1 } ) \right] = 0 } \\ { s _ { 1 } ( \| a \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \gamma _ { 1 } ) = 0 , s _ { 1 } \geq 0 } \end{array} \right. } \end{array}
$$

The first condition reveals that $s _ { 1 } \neq 0$ and $a = s _ { 1 } ^ { - 1 } \bar { \Sigma } _ { k } \hat { \Sigma } _ { k } ^ { - 1 } b$ thus the second condition leads to $\| a \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } = \gamma _ { 1 }$ . Then,

$$
s _ { 1 } = \frac { \| \bar { \Sigma } _ { k } \hat { \Sigma } _ { k } ^ { - 1 } b \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } } { \| a \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } } = \frac { \| \bar { \Sigma } _ { k } \hat { \Sigma } _ { k } ^ { - 1 } b \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } } { \sqrt { \gamma _ { 1 } } } ,
$$

and the proof is completed.

Substituting the optimal $a ^ { * }$ into the problem，the inner optimization of (12) can be further reformulated as a convex maximization problem:

$$
\operatorname* { m a x } _ { b } \alpha \| b \| _ { A } + \| b \| _ { B } ^ { 2 } \mathrm { ~ s . t . ~ } \| b \| _ { 2 } ^ { 2 } \leq \gamma _ { 0 } ( z ) ,
$$

where $A = \hat { \Sigma } _ { k } ^ { - 1 } \bar { \Sigma } _ { k } \hat { \Sigma } _ { k } ^ { - 1 }$ ， $B = \hat { \Sigma } _ { k } ^ { - 1 }$ ,and $\alpha = 2 \sqrt { \gamma _ { 1 } }$

# $D$ Complexity Analysis

As we have shown, solving the one-step DRPP ${ \bf P } _ { 1 }$ is equivalent to finding a global minimax point (Stackelberg equilibrium） for a convex-nonconcave minimax problem where the outer minimization is a semidefinite programming and the inner problem is to maximize a non-smooth convex function subject to a quadratic constraint. It is known that convex maximization is NP-hard in very simple cases,such as quadratic maximization over a hypercube,and even verifying local optimality is NP-hard [41]. Thus, finding a global minimax point of a one-step DRPP problem is also NP-hard.

Even worse,finding a local minimax point of a general constrained minimax optimization is of fundamental hardness. Not only may the local minimax point not exist [42], but any gradient-based algorithm needs exponentially many queries in the dimension and $\epsilon ^ { - 1 }$ to compute an $\epsilon$ approximate first-order local stationary point [43]. Finally, the local stationary point is not always guaranteed to be a local minimax point [44], [45].

In summary，globally finding a minimax point of a onestep DRPP is computationally intractable,and gradient-based algorithms aiming for local minimax points are limited by both convergence speed and performance guarantee. Rather than pursuing the optimal solutions of $\mathbf { P } _ { 0 }$ ,a more reasonable goal is to derive suboptimal solutions for the one-step DRPP $\mathbf { P } _ { 1 }$ .Based on these suboptimal solutions,we can then derive upperadowrbounsfortbustauefcos $V _ { k } ^ { * }$

# VI. VALUE FUNCTION UPPER BOUND

To get an upper bound of $V _ { k } ^ { * } ( z )$ ， one can either enlarge the feasible region of the outer maximization or shrink the feasible region of the inner minimization.In this paper, we choose to shrink the ambiguity set $\mathcal { T } _ { k } ( z )$ . If the first constraint is strengthened as $\nu _ { k } = \bar { \nu } _ { k }$ ，one is constrained to a smaller ambiguity subset of $\mathcal { T } _ { k } ( z )$ . If a global minimax point can be solved for the relaxed one-step DRPP,one will get a suboptimal predictor and a performance upper bound.

# A．Problem Reformulation

If we shrink the ambiguity set of one-step DRPP $\mathbf { P } _ { 1 }$ by assuming $\overline { { \nu } } _ { k } = \nu _ { k }$ ，there is ${ \pmb w } _ { k } = { \pmb x } _ { k + 1 } - { \bar { \nu } _ { k } }$ ，which means that ${ \pmb w } _ { k }$ can be precisely known after $\scriptstyle { \pmb { x } } _ { k + 1 }$ been observed. Now that there is no uncertainty of the state evolution $\nu _ { k }$ ， predicting the state's conditional probability measure $P _ { \mathbf { \boldsymbol { x } } _ { k + 1 } | \mathbf { \boldsymbol { z } } _ { k } }$ isequivalenttopredictingtheoiseprobabilityesure $P _ { w _ { k } }$ ， and the one-step DRPP ${ \bf P } _ { 1 }$ is then reformulated as

$$
\begin{array} { r l } & { \underset { \ b { \hat { p } } _ { k } \in \mathcal { F } ^ { \mu _ { k } , \hat { \rho } _ { k } } } { \operatorname* { s u p } } \ \underset { \ b { \hat { \mathcal { p } } } _ { k } } { \operatorname* { i n f } } \ \int _ { \mathcal { X } } \log [ \hat { p } _ { k } ( \boldsymbol { w } + \boldsymbol { \bar { \nu } } _ { k } ) ] \mathrm { d } \tilde { \rho } _ { k } ( \boldsymbol { w } ) } \\ & { \mathrm { s . t . } \left\{ \begin{array} { l } { \hat { \rho } _ { k } \in \mathcal { M } _ { + } ( \mathcal { X } ) , \mathbb { E } _ { \boldsymbol { w } \sim \tilde { \rho } _ { k } } [ 1 ] = 1 , \mathbb { E } _ { \boldsymbol { w } \sim \tilde { \rho } _ { k } } [ \boldsymbol { w } ] = \mu _ { k } } \\ { \gamma _ { 3 } \boldsymbol { \bar { \Sigma } } _ { k } \preceq \mathbb { E } _ { \boldsymbol { w } \sim \tilde { \rho } _ { k } } \left[ \left( \boldsymbol { w } - \boldsymbol { \bar { \mu } } _ { k } \right) \left( \boldsymbol { w } - \boldsymbol { \bar { \mu } } _ { k } \right) ^ { \top } \right] \preceq \gamma _ { 2 } \boldsymbol { \bar { \Sigma } } _ { k } } \\ { ~ \left[ ~ \begin{array} { c } { \boldsymbol { \bar { \Sigma } } _ { k } } \\ { \left( \mu _ { k } - \boldsymbol { \bar { \mu } } _ { k } \right) ^ { \top } } \\ { \gamma _ { 1 } } \end{array} \right. ~ \left. \begin{array} { c } { \left( \mu _ { k } - \boldsymbol { \bar { \mu } } _ { k } \right) } \\ { \gamma _ { 1 } } \end{array} \right] \succeq 0 . } \end{array} \right. } \end{array}
$$

With the outer maximization imposed on $\hat { p } _ { k }$ ，we can get a reformulated one-step DRPP whose optimal worst-case value function is an upper bound of the objective of ${ \bf P } _ { 1 }$

# B.Noise-DRPP

Taking the dual form of the inner problem of（15）and joining it with the outer maximization, we transform the primal maximin problem into the following maximization problem:

$$
\begin{array} { r l } & { ( \mathbf { D } _ { 2 } ) : \underset { \hat { \boldsymbol { p } } _ { k } \in \mathcal { F } , \boldsymbol { r } , \boldsymbol { q } , \boldsymbol { Q } _ { 1 } , \boldsymbol { Q } _ { 2 } , \boldsymbol { P } , \boldsymbol { p } , \boldsymbol { s } } G ( \boldsymbol { r } , \boldsymbol { q } , \boldsymbol { Q } _ { 1 } , \boldsymbol { Q } _ { 2 } , \boldsymbol { P } , \boldsymbol { p } , \boldsymbol { s } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \mathrm { s . t . } \left\{ \begin{array} { l } { w ^ { \top } ( \boldsymbol { Q } _ { 1 } - \boldsymbol { Q } _ { 2 } ) w + w ^ { \top } \boldsymbol { q } + \boldsymbol { r } + \log \hat { p } _ { k } ( w + \bar { \boldsymbol { \nu } } _ { k } ) \geq 0 } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \forall w \in \mathbb { R } ^ { d _ { x } } } \\ { \boldsymbol { q } + 2 ( \boldsymbol { Q } _ { 1 } - \boldsymbol { Q } _ { 2 } ) \bar { \mu } _ { k } + 2 \boldsymbol { p } = 0 } \\ { \boldsymbol { Q } _ { 1 } \succeq 0 , \boldsymbol { Q } _ { 2 } \succeq 0 } \\ { \left[ \begin{array} { l l } { \boldsymbol { P } } & { \boldsymbol { p } } \\ { \boldsymbol { p } ^ { \top } } & { \boldsymbol { s } } \end{array} \right] \succeq 0 , } \end{array} \right. } \end{array}
$$

where $G ( r , q , Q _ { 1 } , Q _ { 2 } , P , p , s ) ~ = ~ - r + { \bar { \mu } _ { k } } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) { \bar { \mu } _ { k } } ~ -$ $( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P ) \cdot \bar { \Sigma } _ { k } + 2 p ^ { \top } \bar { \mu } _ { k } - s \gamma _ { 1 }$ . Please see Appendix F for a detailed derivation of the dual problem.

Theorem 3 (Noise-DRPP). Given $z _ { k } = z \in \mathcal { Z }$ at time step $k \in \{ 0 , \ldots , T - 1 \}$ ，if the probabilistic predictor has no ambiguity of the one-step state evolution, i.e., $\bar { \nu } _ { k } = \nu _ { k }$ ，the solution to the one-step DRPP $\mathbf { P } _ { 1 }$ is:   
$i )$ The optimal predictive pdf is

$$
\begin{array} { r } { \hat { p } _ { k } ^ { * } \sim \mathcal { N } \left( \bar { \nu } _ { k } + \bar { \mu } _ { k } , \gamma _ { 2 } \bar { \Sigma } _ { k } \right) . } \end{array}
$$

$\romannumeral 2$ ） The worst-case conditional measure $\rho _ { k } ^ { * }$ belongs to the set

$$
\left\{ P _ { \pmb { x } _ { k + 1 } | \mathscr { z } _ { k } } ( \cdot \mid \boldsymbol { z } ) \begin{array} { l }  \left| \begin{array} { l } { \pmb { x } _ { k + 1 } = \bar { f } _ { k } ( \boldsymbol { z } ) + \pmb { w } _ { k } } \\ { \mathbb { E } _ { \boldsymbol { w } \sim P _ { \boldsymbol { w } _ { k } } } [ ( \boldsymbol { w } - \bar { \mu } _ { k } ) ( \boldsymbol { w } - \bar { \mu } _ { k } ) ^ { \top } ] = \gamma _ { 2 } \bar { \Sigma } _ { k } } \end{array} \right. \right\} . \end{array}
$$

ii) The objective function at $( \hat { p } _ { k } ^ { * } , \rho _ { k } ^ { * } )$ is

$$
- \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + d _ { x } + \log \mathrm { d e t } ( \gamma _ { 2 } \bar { \Sigma } _ { k } ) \right] .
$$

Proof. Please see Appendix $\mathrm { G }$

First,an immediate application of Theorem 3 is to utilize the explicit solution of $\hat { p } _ { k } ^ { * }$ to develop a probabilistic prediction algorithm.Because this predictor solely focuses on the distributional robustness against the ambiguity of system noises, we name it Noise-DRPP. The pseudo-code is in Algorithm 1.

Second,notice that the optimal predictive pdf is unique, but the worst-case SDS is not.It contains any SDS that has a proper one-step state evolution and a noise whose first two moments satisfy an equation.

Finally,which is also our original goal, the objective function at $( \hat { p } _ { k } ^ { * } , \rho _ { k } ^ { * } )$ is an upper bound of the optimal objective of ${ \bf P } _ { 1 }$ . Since the one-step upper bound does not depend on $z$ ， one can recursively use the Bellman equation （7） to get an upper bound of the robust value function $V _ { k } ^ { * }$

Theorem 4 (Upper bound of $V _ { k } ^ { * }$ ).Given an initial statecontrol pair $z \in { \mathcal { Z } }$ at time step $k \in \{ 0 , \ldots , T - 1 \}$ ，an upper bound of the robust value function $V _ { k } ^ { * } ( z )$ is

$$
V _ { k } ^ { * } ( z ) \leq \sum _ { t = k } ^ { T - 1 } - \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + d _ { x } + \log \operatorname* { d e t } ( \gamma _ { 2 } \bar { \Sigma } _ { t } ) \right] .
$$

Remark 2. We highlight that this upper bound can be computed ofline because it only depends on the parameters of ambiguity sets $\mathcal { T } _ { 0 : T - 1 }$ . Furthermore, the conservatism of this upper bound can be evaluated by the gap between it and another lower bound, as developed in the next section.

# VII. VALUE FUNCTION LOWER BOUND

To get a lower bound of $V _ { k } ^ { * } ( z )$ ，one can either shrink the feasible region of the outer maximization or enlarge the feasible region of the inner minimization. Because enlarging the ambiguity set does not essentially change the structure to alleviate the difficulty of a one-step DRPP $\mathbf { P } _ { 2 }$ ，we choose to restrict $\hat { p } _ { k }$ to a smaller pdf family by forcing the eigenvectors of the predictive covariance $\hat { \Sigma } _ { k }$ to be the same as the nominal covariance $\bar { \Sigma } _ { k }$ . In this section, we elaborate on how the eigenvector restriction contributes to a well-performed algorithm and a lower bound.

Input: time horizon $T$ ,ambiguity sets $\mathcal { I } _ { 0 : T - 1 }$ , control policy   
$\pi _ { 0 : T - 1 }$ , initial state $x _ { 0 }$   
1: $s _ { 0 } \gets 0$ > score at time step O   
2: for $k = 0 , \ldots , T - 1$ do   
3: Predictor updates ambiguity set $\mathcal { T } _ { k }$ as (3).   
4: SDS generates control input $u _ { k } \sim \pi _ { k } ( \cdot \mid x _ { k } )$   
5: $z _ { k } \gets ( x _ { k } , u _ { k } )$ ， $\bar { \nu } _ { k }  \bar { f } _ { k } \bar { ( z _ { k } ) }$   
6: Predictor predicts $\hat { p } _ { k } ^ { * } \sim \mathcal { N } \left( \bar { \nu } _ { k } + \bar { \mu } _ { k } , \ \gamma _ { 2 } \bar { \Sigma } _ { k } \right)$   
7: SDS generates the next state $x _ { k + 1 }$   
8: $s _ { k + 1 } \gets s _ { k } + \mathcal { L } ( \hat { p } _ { k } ^ { * } , x _ { k + 1 } )$

9: end for

Output: states $x _ { 0 : T }$ ， predictions $\hat { p } _ { 0 : T - 1 } ^ { * }$ ，scores $s _ { 0 : T }$

# A． Problem Reformulation

By adding auxiliary constraints to the predictor in (12), one can still get a suboptimal prediction performance guarantee in the worst case,which is also a performance lower bound to the original one-step DRPP. Nevertheless,the choice of relaxation methods is delicate,which can greatly affect both the optimality gap and the computation efficiency.

Suppose the spectral decomposition for $\bar { \Sigma } _ { k }$ is $Q _ { k } \Lambda _ { k } Q _ { k } ^ { \top }$ where $\begin{array} { r l r } { Q _ { k } } & { { } = } & { \left[ { \bf v } _ { 1 , k } , \ldots , \ldots , { \bf v } _ { d _ { x } , k } \right] } \end{array}$ isan orthogonal matrix whose columns are eigenvectors of $\bar { \Sigma } _ { k }$ and $\begin{array} { r l } { \Lambda _ { k } } & { { } = } \end{array}$ $\mathrm { d i a g } \{ \lambda _ { 1 , k } , . . . , \lambda _ { d _ { x } , k } \}$ . Similarly， we let the spectral decomposition for $\hat { \Sigma } _ { k }$ be $\hat { Q } _ { k } \hat { \Lambda } _ { k } \hat { Q } _ { k } ^ { \top }$ ，where $\hat { Q } _ { k }$ is an orthogonal matrix whose columns are eigenvectors of $\hat { \Sigma } _ { k }$ and $\boldsymbol { \tilde { \Lambda _ { k } } } \ =$ $\mathrm { d i a g } \{ \hat { \lambda } _ { 1 , k } , \dotsc , \hat { \lambda } _ { d _ { x } , k } \}$ . Then，the eigenvector restriction for (12) is to impose the following constraint on the predictor:

$$
{ \hat { Q } } _ { k } = Q _ { k } .
$$

To explain why we chose the eigenvector restriction，some supporting lemmas need to be derived.

Lemma 3. The optimal $b ^ { * }$ to problem (14) is an eigenvector $\frac { \alpha } { \| b ^ { * } \| _ { A } } A + 2 B$ with is $\ell ^ { 2 }$ -norm being $\sqrt { \gamma _ { 0 } ( z ) }$

Proof. There are necessary optimality conditions for (14):

Proof. When $\hat { Q } _ { k } = Q _ { k }$ is satisfied, there is

$$
\frac { \alpha } { \| b \| _ { A } } A + 2 B = Q _ { k } \left[ \frac { \alpha } { \| b \| _ { A } } \hat { \Lambda } _ { k } ^ { - 2 } \Lambda _ { k } + \hat { \Lambda } _ { k } ^ { - 1 } \right] Q _ { k } ^ { \top } ,
$$

whose eigenvectors are the columns of $Q _ { k }$ . Lemma 3 indicates that the optimal $b$ can be expressed as $\sqrt { \gamma _ { 0 } ( z ) } \mathbf { v } _ { i }$ for certain $\begin{array} { r l r } { i } & { { } \in } & { \{ 1 , \ldots , d _ { x } \} , } \end{array}$ then we have the objective of (14) is $( 2 \sqrt { \gamma _ { 0 } ( z ) \gamma _ { 1 } \lambda _ { i , k } } + \gamma _ { 0 } ( z ) ) \hat { \lambda } _ { i , k } ^ { - 1 }$ Since $j _ { k } ~ =$ ar $\mathrm { g } \operatorname* { m a x } _ { i } ( 2 \sqrt { \gamma _ { 0 } ( z ) \gamma _ { 1 } \lambda _ { i , k } } + \gamma _ { 0 } ( z ) ) \hat { \lambda } _ { i , k } ^ { - 1 }$ problem (14) is solved as $b ^ { * } = \sqrt { \gamma _ { 0 } ( z ) } \mathbf { v } _ { j _ { k } }$ · □

Utilizing the expression of $b ^ { * }$ in Lemma 4,the minimax optimization （12） under eigenvector restriction (16） can be transformed into

$$
\begin{array} { r l r } {  { ( { \bf P } _ { 3 } ) : \sum _ { \hat { \Lambda } _ { k } \succeq 0 , j _ { k } } \sum _ { i = 1 } ^ { d _ { x } } [ - \log ( \hat { \lambda } _ { i , k } ^ { - 1 } ) + \gamma _ { 2 } \lambda _ { i , k } \hat { \lambda } _ { i , k } ^ { - 1 } ] + } } \\ & { } & { ( 2 \sqrt { \gamma _ { 0 } ( z ) \gamma _ { 1 } \lambda _ { j _ { k } , k } } + \gamma _ { 0 } ( z ) ) \hat { \lambda } _ { j _ { k } , k } ^ { - 1 } } \\ & { } & { \mathrm { s . t . } \ \hat { \lambda } _ { j _ { k } , k } \leq \frac { 2 \sqrt { \gamma _ { 0 } ( z ) \gamma _ { 1 } \lambda _ { j _ { k } , k } } + \gamma _ { 0 } ( z ) } { 2 \sqrt { \gamma _ { 0 } ( z ) \gamma _ { 1 } \lambda _ { i , k } } + \gamma _ { 0 } ( z ) } \hat { \lambda } _ { i , k } } \\ & { } & { \forall i \in \{ 1 , \ldots , d _ { x } \} . } \end{array}
$$

# B.Eig-DRPP

Notice that optimization $\mathbf { P } _ { 3 }$ is convex for any fixed $j _ { k }$ Therefore, $\mathbf { P } _ { 3 }$ can be solved by taking the minimum of $d _ { x }$ different convex optimization problems,which is numerically efficient. Let $\left( \hat { \Lambda } _ { k } ^ { * } = \mathrm { d i a g } \{ \hat { \lambda } _ { 1 , k } ^ { * } , \ldots , \hat { \lambda } _ { d _ { x } , k } ^ { * } \} , j _ { k } ^ { * } \right)$ be the solution of $\mathbf { P } _ { 3 }$ ，we summarize the solution of a one-step DRPP under eigenvector restriction as follows.

Theorem 5 (Eig-DRPP). Given $z _ { k } = z \in \mathcal { Z }$ at time step $k \in \{ 0 , \ldots , T - 1 \}$ ,if the probabilistic predictor is constrained by the eigenvector restriction (16), the solution to the one-step $D R P P \ \mathbf { P } _ { 1 }$ is:

$i )$ The optimal predictive pdf $\hat { p } _ { k } ^ { * }$ is

$$
\left\{ \begin{array} { l l } { \left( \displaystyle \frac { \alpha } { \| b \| _ { A } } A + 2 B - 2 s I \right) b = 0 } \\ { \| b \| _ { 2 } ^ { 2 } = \gamma _ { 0 } ( z ) , } \end{array} \right.
$$

$$
\begin{array} { r } { \hat { p } _ { k } ^ { * } \sim \mathcal { N } \left( \bar { \nu } _ { k } + \bar { \mu } _ { k } , \ Q _ { k } \hat { \Lambda } _ { k } ^ { * } Q _ { k } ^ { \top } \right) . } \end{array}
$$

where the first one comes from the KKT condition and the second one holds because the objective increases monotonously with $\| \boldsymbol { b } \| _ { 2 }$ . It follows that the optimal $b$ should be an eigenvector of ${ \frac { \alpha } { \| b \| _ { A } } } A + 2 B$ with its $\ell ^ { 2 }$ -norm being $\sqrt { \gamma _ { 0 } ( z ) }$ □

If the eigenvectors of $\frac { \alpha } { \parallel b ^ { * } \parallel _ { A } } A + 2 B$ have explicit expressions,one may have explicit expressions for $a ^ { * } , b ^ { * }$ . Then, substituting these expressions into (12) can transform the original minimax optimization problem into a semidefinite-constrained maximization,where tractable optimization solvers may be available.However,it is always difficult to explicitly derive the eigenvectors when $\hat { Q } _ { k }$ does not identify with $Q _ { k }$

Lemma 4. When $\hat { Q } _ { k } = Q _ { k }$ is supplemented to the constraints of (14), the solution is

$$
b ^ { * } = \sqrt { \gamma _ { 0 } ( z ) } \mathbf { v } _ { j _ { k } , k } ,
$$

ii）The worst-case conditional measure $\rho _ { k } ^ { * }$ belongs to the set

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathbf { x } _ { k + 1 } = f _ { k } ( z ) + \mathbf { w } _ { k } } \\ { f _ { k } ( z ) = \sqrt { \gamma _ { 0 } ( z ) } \mathbf { v } _ { j _ { k } ^ { * } , k } + \bar { \nu } _ { k } } \\ { \mathbb { E } [ \mathbf { w } _ { k } ] = \bar { \mu } _ { k } + \sqrt { \gamma _ { 1 } \lambda _ { j _ { k } ^ { * } , k } } \mathbf { v } _ { j _ { k } ^ { * } , k } } \\ { \mathrm { C o v } [ \mathbf { w } _ { k } ] = \gamma _ { 2 } \bar { \Sigma } _ { k } - \gamma _ { 1 } \lambda _ { j _ { k } ^ { * } , k } \mathbf { v } _ { j ^ { * } , k } \mathbf { v } _ { j _ { k } ^ { * } , k } ^ { \top } } \end{array} \right\} . } \end{array}
$$

ii) The objective function at $( \hat { p } _ { k } ^ { * } , \rho _ { k } ^ { * } )$ is

$$
\begin{array} { c } { { \displaystyle - \frac { 1 } { 2 } \left\{ d _ { x } \log ( 2 \pi ) + \sum _ { i = 1 } ^ { d _ { x } } \left[ \log ( \hat { \lambda } _ { i , k } ^ { * } ) + \gamma _ { 2 } \frac { \lambda _ { i , k } } { \hat { \lambda } _ { i , k } ^ { * } } \right] \right. } } \\ { { \displaystyle \left. + \frac { 2 \sqrt { \gamma _ { 0 } ( z ) } \gamma _ { 1 } \lambda _ { j _ { k } ^ { * } , k } + \gamma _ { 0 } ( z ) } { \hat { \lambda } _ { j _ { k } ^ { * } , k } ^ { * } } \right\} . } } \end{array}
$$

Proof. The proof is completed by combining the results in Lemma 1,Lemma 2,and the solutions of $\mathbf { P } _ { 3 }$ □

First,an immediate application of Theorem 5 is to utilize the explicit solution of $\hat { p } _ { k } ^ { * }$ to develop a probabilistic prediction algorithm, named Eig-DRPP. The pseudo-code is presented in Algorithm 2. Second,although the optimal predictive pdf is unique,the worst-case conditional measure is not. Compared to Theorem 3,there are relatively more restrictions on the set of worst-case conditional measures.Third,when $\gamma _ { 0 } = 0$ ， i.e., the predictor has no ambiguity of $f _ { k }$ , the solution of $\mathbf { P } _ { 3 }$ is $\hat { \lambda } _ { i , k } ^ { * } \doteq \gamma _ { 2 } \lambda _ { i , k }$ and $j _ { k } ^ { * }$ can be any feasible index. In this case,both the optimal predictive pdf and objective value in Theorem 5 coincide with those in Theorem (3).However, the sets of worst-case conditional measures are not the same, due to the extra constraint for the expectation in Theorem 5.

Finally, the objective function at $( \hat { p } _ { k } ^ { * } , \rho _ { k } ^ { * } )$ is a lower bound of the optimal objective of ${ \bf P } _ { 1 }$ . Since the objective value depends on $z$ through $\gamma _ { 0 } ( z )$ ，one cannot directly use the Bellman equation (7)recursively to get a lower bound of $V _ { k } ^ { * }$

Theorem 6 (Lower bound of $V _ { k } ^ { * }$ and optimality gap of $\mathcal { F } _ { \mathrm { E i g } } )$ Given an initial state-control pair $z \in { \mathcal { Z } }$ at time step $k \in$ $\{ 0 , \ldots , T - 1 \} ,$ ， and an upper bound of $\gamma _ { 0 }$ such that $\gamma _ { 0 } ( z ) \leq$ $\Gamma _ { 0 } \in \mathbb { R } _ { + } , \forall z \in \mathcal { Z }$ ，there is $i )$ A lower bound of the robust value function $V _ { k } ^ { * } ( z )$ is

$$
V _ { k } ^ { * } ( z ) \geq \sum _ { t = k } ^ { T - 1 } - \frac { 1 } { 2 } \left\{ d _ { x } \log ( 2 \pi ) + \sum _ { i = 1 } ^ { d _ { x } } \left[ \log ( \hat { \lambda } _ { i , t } ^ { * } ) + \gamma _ { 2 } \frac { \lambda _ { i , t } } { \hat { \lambda } _ { i , t } ^ { * } } \right] \right.
$$

where $\hat { \lambda } _ { 1 , t } ^ { * } , \ldots , \hat { \lambda } _ { d _ { x } , t } ^ { * } , j _ { t } ^ { * }$ is the solution to $\mathbf { P } _ { 3 }$ with $k$ being replaced by $k$ and all $\gamma _ { 0 } ( z )$ being replaced by $\Gamma _ { 0 }$ $i i$ )The optimality gap between Eig-DRPP $\mathcal { F } _ { E i g }$ and the global optimal predictor is upper bounded as follows,

$$
\begin{array} { r l } & { V _ { k } ^ { * } ( z ) - V _ { k } ^ { \mathcal { F } _ { E i g } } ( z ) \leq \displaystyle \sum _ { t = k } ^ { T - 1 } - \frac { 1 } { 2 } \Bigg \lbrace \underbrace { d _ { x } + \log \operatorname* { d e t } ( \gamma _ { 2 } \bar { \Sigma } _ { t } ) } _ { \mathscr { O } } } \\ & { - \displaystyle \sum _ { i = 1 } ^ { d _ { x } } \left[ \log ( \hat { \lambda } _ { i , t } ^ { * } ) + \gamma _ { 2 } \frac { \lambda _ { i , t } } { \hat { \lambda } _ { i , t } ^ { * } } \right] - \frac { 2 \sqrt { \Gamma _ { 0 } \gamma _ { 1 } \lambda _ { j _ { t } ^ { * } , t } } + \Gamma _ { 0 } } { \hat { \lambda } _ { j _ { t } ^ { * } , t } ^ { * } } \Bigg \rbrace . } \end{array}
$$

Proof. i) Notice that as $\gamma _ { 0 } ( z )$ increases,the ambiguity set enlarges, thus the optimal value of one-step DRPP under eigenvector constraint should monotonously decrease. Since the upper bound of $\gamma _ { 0 } ( z )$ ,i.e., $\Gamma _ { 0 }$ ,is independent of $z$ ，one can still use(7) to derive a lower bound.

ii） Subtracting this lower bound from the upper bound in Theorem 4,one immediately gets an upper bound of the optimality gap for Eig-DRPP. As illustrated in (17), $\textcircled{1}$ comes from the upper bound,which is determined by $\gamma _ { 2 }$ and the determinant of $\textstyle { \overline { { \Sigma } } } _ { t }$ .While $\textcircled{2}$ comes from the lower bound, which is jointly determined by $\gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 }$ and the eigenvalues of $\textstyle { \overline { { \Sigma } } } _ { t }$ by solving $\mathbf { P } _ { 3 }$ □

The proposed DRPP predictors can be naturally integrated into SMPC.In particular, the Noise-DRPP predictor provides a closed-form Gaussian predictive model that can be used to construct distributionally robust chance constraints. The Eig-DRPP predictor,while computationally more involved, can serve as an offline surrogate to characterize the trade-off between robustness and conservatism in SMPC design.

# Algorithm 2 Eig-DRPP FEig

[nput: time horizon $T$ , ambiguity sets $\mathcal { T } _ { 0 : T - 1 }$ , control policy   
$\pi _ { 0 : T - 1 }$ ,initial state $x _ { 0 }$   
1: $s _ { 0 } \gets 0$ score at time step O   
2:for $k = 0 , \ldots , T - 1$ do   
3: Predictor updates ambiguity set $\mathcal { I } _ { k }$ as (3).   
4: Do spectral decomposition for $\bar { \Sigma } _ { k } = Q _ { k } \Lambda _ { k } Q _ { k } ^ { \top }$   
5: SDS generates control input $u _ { k } \sim \pi _ { k } ( \cdot \mid x _ { k } )$   
6: $z _ { k } \gets ( x _ { k } , u _ { k } )$ ， $\bar { \nu } _ { k }  \bar { f } _ { k } \bar { ( z _ { k } ) }$   
7: $j _ { k } ^ { * } \gets 0$ ， $\mathrm { v a l } ^ { * }  \infty$ ， $\hat { \Lambda } _ { k } ^ { * } \gets \Lambda _ { k }$   
8: for $j = 1 , 2 , \ldots , d _ { x }$ do   
9: Fix $j _ { k } = j$ solve $\mathbf { P } _ { 3 }$ where te optimizer is $\hat { \Lambda } _ { k } ^ { \left( j \right) }$   
and the optimal value is $\mathrm { v a l } ^ { ( j ) }$ . For each $j$ ,this step is   
a convex optimization   
10: if $\begin{array} { r l } & { \mathrm { ~ \gamma _ { v a l } ^ { \mathtt { i } } ( \mathcal { j } ) < v a l ^ { * } ~ t h e n ~ } } \\ & { \quad \ j _ { k } ^ { * } \gets j , \mathrm { ~ v a l ^ { * } \gets v a l ^ { ( j ) } , ~ \hat { \Lambda } _ { k } ^ * \gets \hat { \Lambda } _ { k } ^ { ( j ) } . } } \end{array}$   
11:   
12: end if   
13: end for   
14: Predictor predicts $\hat { p } _ { k } ^ { * } \sim \mathcal { N } \left( \bar { \nu } _ { k } + \bar { \mu } _ { k } , \ Q _ { k } \hat { \Lambda } _ { k } ^ { * } Q _ { k } ^ { \top } \right)$   
15: SDS generates the next state $x _ { k + 1 }$   
16: $s _ { k + 1 } \gets s _ { k } + \mathcal { L } ( \hat { p } _ { k } ^ { * } , x _ { k + 1 } )$

17: end for

Output: states $x _ { 0 : T }$ ， predictions $\hat { p } _ { 0 : T - 1 } ^ { * }$ ， scores $s _ { 0 : T }$

# VIII. EXPERIMENTS

In this section,a series of experiments is conducted to explore three questions.i) Given an SDS subject to an ambiguity set, what are the performance advantages of NoiseDRPP and Eig-DRPP compared to a nominal predictor? ii) How much will the prediction performances be influenced by control strategies? ii) How can DRPP predictors be applied to providing high probability confidence regions of future states?

# A. Experiment Setup

Ambiguity set: Conditioned on a state-control pair $z _ { k }$ at time step $k \in \{ 0 , 1 , \cdots , 3 1 \}$ ，the nominal state evolution function is given as

$$
{ \bar { f } } _ { k } ( { \pmb x } _ { k } , { \pmb u } _ { k } ) = \left( \begin{array} { c c } { { 1 } } & { { 0 . 1 } } \\ { { 0 } } & { { 1 } } \end{array} \right) { \pmb x } _ { k } + \left( \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { 0 } } & { { 1 } } \end{array} \right) { \pmb u } _ { k } ,
$$

and the uncertainty between $\boldsymbol { \bar { f _ { k } } }$ and the true $f _ { k }$ is quantified by $\gamma _ { 0 } ( z _ { k } ) ~ = ~ \operatorname* { m i n } \{ 0 . 3 \| z _ { k } \| _ { 2 } , 5 \} ^ { 2 }$ . The nominal mean and covariance of the noise ${ \pmb w } _ { k }$ conditioned on $z _ { k }$ are $\begin{array} { r l } { \overline { { \mu } } _ { k } } & { { } = } \end{array}$ $\binom { 0 } { 0 } , \bar { \Sigma } _ { k } = \binom { 1 } { 0 . 5 } \ 1 . 5 $ ，and the uncertainty between $\bar { \mu } _ { k } , \bar { \Sigma } _ { k }$ and the real $\dot { \mu } _ { k } , \Sigma _ { k }$ are quantified by $\gamma _ { 1 } = 0 . 5$ and $\gamma _ { 2 } = 3$ respectively.Given the ambiguity set specified above, the real SDS belongs to $\mathcal { T } _ { k }$ ． Although the nominal model is linear time-invariant (LTI), the real model does not have to be LTI.

![](images/ecf2881116124983b88ed8182e4cc9a2be2380431152cc4df83d04c5d294474c.jpg)  
Fig.2: Prediction performance of diferent probabilistic predictors on different SDSs under different control strategies

Simulation mechanism: Three different simulation mechanisms are conducted to generate the underlying SDSs.

i) Let $\alpha _ { 1 } , \alpha _ { 2 }$ be two independent random variables subject to the uniform distribution on $[ - 1 , 1 ]$ ,and let $\alpha _ { 3 }$ be another independent random variable subject to the uniform distribution on $[ - 1 , 1 ] ^ { 2 }$ . We randomly generates $\alpha _ { i } \sim p _ { \alpha _ { i } }$ for $i = { 1 , 2 , 3 }$ then we simulate the SDS as

$$
\pmb { x } _ { k + 1 } = \left( \begin{array} { c c } { 1 } & { 0 . 1 + 0 . 3 \alpha _ { 1 } } \\ { 0 } & { 1 } \end{array} \right) \pmb { x } _ { k } + \left( \begin{array} { c c } { 1 } & { 0 . 3 \alpha _ { 2 } } \\ { 0 } & { 1 } \end{array} \right) \pmb { u } _ { k } + \pmb { w } _ { k } ,
$$

where $\pmb { w } _ { k } \sim \mathcal { N } \left( 0 . 5 \alpha _ { 3 } , 3 \bar { \Sigma } _ { k } - 0 . 2 5 \alpha _ { 3 } \alpha _ { 3 } ^ { \top } \right)$

ii)For each $i = { 1 , 2 , 3 }$ let $\{ \alpha _ { i , k } \} _ { k = 0 } ^ { 3 1 }$ be random variables that are independent and identically distributed to $p _ { \alpha _ { i } }$ .We randomly generates $\alpha _ { i , k } \sim p _ { \alpha _ { i , k } }$ for each $i$ and $k$ ，then we simulate the SDS as

$$
\pmb { x } _ { k + 1 } = \left( \begin{array} { c c } { 1 } & { 0 . 1 + 0 . 3 \alpha _ { 1 , k } } \\ { 0 } & { 1 } \end{array} \right) \pmb { x } _ { k } + \left( \begin{array} { c c } { 1 } & { 0 . 3 \alpha _ { 2 , k } } \\ { 0 } & { 1 } \end{array} \right) \pmb { u } _ { k } + \pmb { w } _ { k } ,
$$

where $\pmb { w } _ { k } \sim \mathcal { N } ( 0 . 5 \alpha _ { 3 , k } , 3 \bar { \Sigma } _ { k } - 0 . 2 5 \alpha _ { 3 , k } \alpha _ { 3 , k } ^ { \top } )$

iii) (Adversarial) In this case,the underlying SDS is allowed to adversarially choose an SDS in the ambiguity set at each step to degrade the prediction performance.Specifically，at each step,after a predictive pdf has been output by the predictor, the adversarial SDS uses the predicted covariance to choose a worst-case SDS as described in Theorem 5.

Predictors comparison: Two different control strategies are considered,the zero input and the linear quadratic regulation (LQR),where the zero input strategy means no control input is imposed,and the LQR control strategy is defined as a standard linear quadratic regulator based on the nominal linear model with $Q$ and $R$ all set as identity matrices. Under each control strategy and SDS scenario, we randomly generate 1,000 trajectories starting from the same initial state $x _ { 0 } =$ [2,1]. At each time step $k$ ,let the nominal predictor predicts $\bar { p } _ { k } \sim \mathcal N ( \bar { \nu } _ { k } + \bar { \mu } _ { k } , \bar { \Sigma } _ { k } )$ ，and the optimal predictor predicts $p _ { k } \ \sim \ { \mathcal N } ( \nu _ { k } + \mu _ { k } , \Sigma _ { k } )$ . Then，we compare the prediction performance among the nominal predictor, Noise-DRPP, EigDRPP, and the optimal predictor. Notice that for the adversarial mechanism, there is no explicitly predefined optimal predictor because the real SDS depends on the predictor. Therefore, we should compare the prediction performance among the nominal predictor, Noise-DRPP,and Eig-DRPP.

# B. Results and Discussions

In Fig.2, the prediction performances of different probabilistic predictors on different SDSs under different control strategies are visualized and compared with each other. For each SDS,we evaluate the temporal average scores of each predictor at each time step on 1,OoO randomly sampled trajectories.At each step,the mean of 1,OoO average scores is plotted as a dot,which is an approximation to the expected average score. The $5 \%$ and $9 5 \%$ percentiles of these average scores are plotted as an error bar,reflecting how concentrated the average scores are.

Performance advantages: Under each setting of SDS and control strategy, both Noise-DRPP and Eig-DRPP possess larger expected average scores at each time step compared to the nominal predictor. Since the ambiguity set for our simulation does not have too large an uncertainty,the scores of both DRPP predictors are close to the optimal predictor. Additionally, both DRPP predictors have more concentrated average scores at each time step. As the uncertainty of the real underlying SDS increases from LTI to adversarial, the performance gap between the DRPP predictors and the nominal predictor increases.

![](images/07100dc154bef9b0fd163a33477989ac7fb5abb4e267406fb530b1beda7353aa.jpg)  
Fig. 3: Predictive $90 \%$ confidence regions of probabilistic predictors for different SDSs under diferent control strategies

Influence of control strategies: When a predictor has no ambiguity about the system's state evolution function, predicting the states is equivalent to predicting the noises.In this case, the choice of control strategies does not affect any probabilistic predictor. However, when a predictor does have ambiguity of the system's state evolution function,different control strategies can greatly influence the performance of a probabilistic predictor. Since the ambiguity set at each step directly depends on the state-input pair, different control strategies will lead to quite different state-input trajectories, thus leading to quite different prediction performances.We can observe this phenomenon by comparing (a-c) with (d-f) in Fig. 2 respectively. An intuitive explanation is that the LQR control strategy regulates the system states towards the neighborhoods of O, where the ambiguity of one-step state evolution is smaller than other states,thus the DRPP predictors perform better.

# C. Application: Predict Robust Confidence Regions

An immediate application of probabilistic prediction is to predict high-probability confidence regions for future states. Since the predictive pdfs of both Noise-DRPP and Eig-DRPP are Gaussian,one can easily derive an elliptical $\beta$ -confidence region for any $\beta \in ( 0 , 1 )$ . In Fig.3,we have visualized the predictive $90 \%$ confidence regions for each predictor at certain time steps of a randomly generated trajectory. For the zero control scenario,we visualize the predictions for time steps $k = 1 , 1 6 , 3 1$ , while for the LQR control scenario, we visualize the prediction for time step $k = 3 1$

Visualizing the behaviors of different probabilistic predictors on a trajectory,one can intuitively explain why DRPP predictors outperform the nominal predictor. The nominal predictor is overly confident,where the predictive pdfs are always too sharp to enclose the future states in their $90 \%$ CRs.In contrast, DRPP predictors have taken the ambiguity of noises into account, whose CRs are more robust to contain the prediction targets most of the time. If we further compare Noise-DRPP with Eig-DRPP, it seems that Eig-DRPP is more intelligent in adaptively changing the ellipses’ shapes.This is because Eig-DRPP has also considered the ambiguity of one-step state evolution,and the eigenvalues of predictive covariances are attained based on minimax optimization.

# D. Extension: Nonlinear System

We consider a 6-degree-of-freedom serial manipulator with revolute joints using the standard Euler-Lagrange rigid-body equations [46]. The coordinates are $q ~ \in ~ \mathbb { R } ^ { 6 }$ and velocities $\dot { q } \in \mathbb { R } ^ { 6 }$ . The continuous-time dynamics are

$$
M ( q ) \ddot { q } + C ( q , \dot { q } ) \dot { q } + g ( q ) = \tau ,
$$

where $M ( q )$ is the inertia matrix, $C ( q , \dot { q } ) \dot { q }$ represents the Coriolis forces, $g ( q )$ is teh gravity vector, and $\tau ~ \in ~ \mathbb { R } ^ { 6 }$ denotes applied joint torques. The mass $m _ { i } ~ = ~ 0 . 8 \mathbf { k g }$ ，link length $l _ { i } = 0 . 5 \mathrm { m }$ ，and link center of mass (COM) distance $l _ { c , i } = 0 . 2 5 \mathrm { m }$ for each link $i = 1 , \ldots , 6$ . We write the full state $x = \left[ \boldsymbol { q } ^ { \top } , \dot { \boldsymbol { q } } ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { 1 2 }$

The nominal model adopts a lightweight numerical approximation that preserves the main physical structures while remaining efficient for repeated DRPP computations.For the inertia matrix,a diagonal-dominant rigid-body approximation

$\begin{array} { r } { M _ { i j } ( q ) = \sum _ { k = \operatorname* { m a x } ( i , j ) } ^ { 6 } m _ { k } \bar { l } _ { c , k } ^ { 2 } } \end{array}$   
coupling surrogate, $C _ { i j } ( q , \dot { q } ) = k _ { c } \left( | \dot { q } _ { i } | + | \dot { q } _ { j } | \right) , k _ { c } = 1 0 ^ { - 3 }$ ， which captures the dominant dependence on joint velocities. The gravitational torque is computed from link COM heights, $\begin{array} { r } { g _ { i } ( q ) \mathbf { \bar { \rho } } = m _ { i } g \sum _ { k = 1 } ^ { i } \bar { l _ { c , k } } \cos ( \sum _ { j = 1 } ^ { k } q _ { j } ) } \end{array}$ j=1 qj)，where the gravitational acceleration $g = 9 . 8 1 \mathrm { m / s ^ { 2 } }$ . Finally, the nominal discrete-time dynamics are obtained by RK4 integration with sampling time $\Delta t = 0 . 1 \mathrm { s }$ ：

$$
x _ { k + 1 } = F \left( x _ { k } , u _ { k } \right) + w _ { k } ,
$$

where $\begin{array} { r c l } { u _ { k } } & { = } & { \tau ( k \Delta t ) } \end{array}$ is the zero-holder of the control input $\tau$ in the time interval $[ k \Delta t , ( k + 1 ) \Delta t )$ and $w _ { k }$ models the additive process noise in the state space.For the conditional concic moment-based ambiguity set, we let $\begin{array} { r l r } { \gamma _ { 0 } ( z ) } & { { } = } & { \operatorname* { m i n } \{ 0 . 3 \| z \| _ { 2 } , 5 \} \Delta t ^ { 2 } } \end{array}$ ， $\gamma _ { 1 } ~ = ~ 0 . 5 \Delta t ^ { 2 }$ ， $\gamma _ { 2 } ~ = ~ 3$ $\overline { { \mu } } _ { k } ~ = ~ \mathbf { 0 } _ { 1 2 \times 1 }$ ，the nominal covariance for each pair $( q _ { i } , \dot { q } _ { i } )$ is $\left( { \begin{array} { l l } { \Delta t ^ { 2 } } & { 0 . 5 \Delta t ^ { 2 } } \\ { 0 . 5 \Delta t ^ { 2 } } & { 1 . 5 \Delta t ^ { 2 } } \end{array} } \right) .$ ，and the full nominal covariance $\bar { \Sigma } _ { k }$ is the blockwise combination of these 6 sub-covariances.We simulate the nonlinear SDS under zero control by adversarially choosing one from the ambiguity set at each step against the predictor. The prediction performances of different probabilistic predictors are visualized and compared in Fig. 4.Similar to the results in Fig.2,both Noise-DRPP and Eig-DRPP outperform the nominal predictor significantly in terms of expected average score and score concentration. This experiment demonstrates the effectiveness and practicality of the proposed DRPP framework on nonlinear SDSs.

![](images/aa06cab33cf712047d1597bc1c7e8ae1e0248488e90963cec2e82b9d13de7687.jpg)  
Fig.4:Prediction performance of different probabilistic predictors on the nonlinear manipulator system.

# IX. CONCLUSION

This paper presented a distributionally robust probabilistic prediction (DRPP） framework to optimize the worst-case prediction performance over a predefined ambiguity set of SDSs.To overcome the inherent intractability of optimizing over function spaces,Bellman equation is developed for the robust value function of DRPP,and optimality conditions are exploited to transform the original problem into Euclidean spaces.While achieving the global optimal solution remains computationally prohibitive，we developed two suboptimal predictors:Noise-DRPP,obtained by relaxing the inner constraints,and Eig-DRPP,derived from relaxing the outer constraints. An explicit upper bound on the optimality gap was established for the proposed predictors.Numerical simulations demonstrated the effectiveness and practicality of the proposed predictors across various SDSs.

# REFERENCES

[1] F. Sévellc and S.S.Drijfhout,“A novel probabilistic forecast system predicting anomalouslywarm 2018-2022 reinforcing the long-term global warming trend,”Nature Communications,vol.9,no.1,p. 3024, Aug. 2018.   
[2]E.Y.Cramer,E.L.Ray,V.K.Lopez, J.Bracher,A. Brennen,A.J. Castro Rivadeneira,A. Gerding,T. Gneiting,K. H. House,Y.Huang et al.,“Evaluation of individual and ensemble probabilistic forecasts of covid-19 mortality in the united states,”Proceedings of the National Academy of Sciences,vol. 119,no.15,2022.   
[3] R.Buizza,“The value of probabilistic prediction,”Atmospheric Science Letters,vol.9,no.2,pp.36-42,2008.   
[4]J.Fisac,A. Bajcsy,S.Herbert,D.Fridovich-Keil,S.Wang, C.Tomlin, and A.Dragan,“Probabilistically safe robot planning with confidencebased human predictions,Robotics:Science and Systems XIV,2018.   
[5] T. Gneiting and M. Katzfuss,“Probabilistic forecasting,” Annual Review of Statisticsand ItsApplication,vol.1,no.1,pp.25-151,014. [6] T.Gneiting and A.E.Raftery,“Strictly proper scoring rules,prediction, and estimation,’Journal of the American Statistical Association,vol. 102,no.477,pp.359-378,Mar. 2007. [7] D.Landgraf,A.Volz,F.Berkel,K. Schmidt，T. Specker,and K.Graichen,“Probabilistic prediction methods for nonlinear systems with application to stochastic model predictive control,”Annual Reviews in Control,vol.56,Jan.2023.   
[8] D.Hinrichsen and A.J. Pritchard,“Uncertain systems,”in Mathematical Systems Theory I:Modelling,State Space Analysis,Stability and RobustnessD.HinrichsenandA.J.Pritchard,Eds.Berlin,Hedeberg: Springer,2005,pp.517-713. [9]K.Schmuidgen,The Moment Problem,ser.Graduate Texts in Mathematics.Cham: Springer International Publishing,2017,vol. 277.   
[10] E.Delage and Y. Ye,“Distributionally robust optimization under moment uncertainty with application to data-driven problems,” Operations Research,Jan.2010.   
[11]P.S.Maybeck,Stochastic Models,Estimation,and Control．Academic press,1982.   
[12] M. Roth and F.Gustafsson,“An efficient implementation of the second order extended kalman filter,” in 14th International Conference on Information Fusion,Jul. 2011, pp. 1-6.   
[13] M. Norgaard,N. K. Poulsen,and O.Ravn,“New developments in state estimation for nonlinear systems,Automatica,vol.36,no.11,pp.1627- 1638,Nov.2000.   
[14]H.M. T.Menegaz,J.Y. Ishihara, G. A. Borges,and A.N. Vargas,“A systematization of the unscented kalman filter theory,”IEEE Transactions on Automatic Control,vol. 60,no.10,pp.2583-2598,Oct.2015.   
[15] M. Simandl and J. Dunik,“Derivative-free estimation methods: New results and performance analysis,”Automatica,vol. 45, no.7,pp.1749- 1757,Jul. 2009.   
[16]H.W. Sorenson andD.L. Alspach,“Recursive bayesian estimation using gaussian sums,” Automatica,vol.7,no.4, pp.465-479,Jul.1971.   
[17]D. Alspach and H. Sorenson,“Nonlinear bayesian estimation using gaussian sum approximations,”IEEE Transactions on Automatic Control, vol.17,no.4,pp.439-448,Aug.1972.   
[18]R. Chen and J. S.Liu,“Mixture kalman filters,” Journal of the Royal Statistical Society Series B: Statistical Methodology,vol. 62, no.3,pp. 493-508,Sep.2000.   
[19]N.Wiener,“The homogeneouschaos,’American Journal of Mathematics,vol. 60,no.4,pp.897-936,1938.   
[20] J.A.Paulson and A．Mesbah,“An efficient method for stochastic optimal control with joint chance constraints for nonlinear systems,” International Journal of Robust and Nonlinear Control,vol.29,no.15, pp.5017-5037,2019.   
[21] S. Särkkä and L. Svensson,Bayesian Filtering and Smoothing,2nd ed., ser. Institute of Mathematical Statistics Textbooks.Cambridge: Cambridge University Press,2023.   
[22] J.Zhang,“Modern monte carlo methods for efficient uncertainty quantification and propagation:A survey,’WIREs Computational Statistics, vol.13,no.5, p. e1539,2021.   
[23] M. Farina,L.Giulioni,and R. Scatolini,“Stochastic linear_model predictive control with chance constraints-a review,”Journal of Process Control, vol. 44, pp.53-67,Aug. 2016.   
[24]A.Mesbah,“Stochastic model predictive control:An overview and perspectivesfor futureresearch"IEEE Control SystemsMagazine, vol.36,no.6,pp.30-44, Dec.2016.   
[25] R.D. McAlister and J.B.Rawlings，“Nonlinear stochastic model predictive control: Existence,measurability,and stochastic asymptotic stability”IEEE Transactionson Automatic Control,vol.68,no.3,pp. 1524-1536,Mar.2023.   
[26] J. Coulson,J. Lygeros,and F.Dorfler,“Distributionally robust chance constrained data-enabledpredictive control.” IEEE Transactionson Automatic Control,vol.67, no.7,pp.3289-3304,Jul.2022.   
[27]P. Coppens and P.Patrinos,“Data-driven distributionally robust mpc for onstrained stochastic systems,” IEEE Control Systems Lettrs,vol.6, pp. 1274-1279.   
[28] C.P.Robert,The Bayesian Choice,ser. Springer Texts in Statistics. New York,NY: Springer,2007.   
[29]D.T.Frazier, W. Maneesoonthorn, G.M.Martin,and B.P.M. McCabe, "Approximate bayesian forecasting” International Journal of Forecasting,vol.35,no.2, pp.521-539,Apr. 2019.   
[30] R.Koenker,“Quantileregression:40 yearson”Annual Review of Economics,vol.9,no.Volume 9,2017,pp.155-176,Aug.2017.   
[31] L.Schlosser,T.Hothorn,R. Staufer,and A. Zeileis,“Distributional regression forests for probabilistic precipitation forecasting in complex terrain’The Annalsof Applied Statistics,vol.13,no.3,pp.1564-1589, Sep. 2019.   
[32] T.Duan,A.Anand,D.Y. Ding,K.K. Thai,S.Basu,A. Ng,and A.Schuler，“NGBoost: Natural gradient boosting for probabilistic prediction,’in Proceedings of the 37th International Conference on Machine Learning. PMLR,Nov.2020,pp.2690-2700.   
[33]D. Salinas,V.Flunkert,J. Gasthaus,and T. Januschowski,“DeepAR: Probabilistic forecasting with autoregressive recurrent networks,”International Journal of Forecasting，vol.36,no.3，pp.1181-1191,Jul. 2020.   
[34]H. Tyralis and G.Papacharalampous,“A review of predictive uncertainty estimation with machine learning,”Artificial Intelligence Review,vol.57, no.4,p.94,Mar.2024.   
[35]H.E. Scarf,K.J. Arrow,and S. Karlin,A Min-Max Solution of an Inventory Problem．Rand Corporation Santa Monica,1957.   
[36] G.Gallgo and I.Moon,“The distribution free newsboy problem: Reviewand extensions" Journal ofthe OperationalResearch Society vol. 44, no.8,pp.825-834,Aug. 1993.   
[37] A.Shapiro and A.Pichler,“Conditional distributionally robust functionals, Operations Research,vol. 72,no.6, pp.2745-2757.   
[38] M. Parry,A. P. Dawid,and S.Lauritzen,“Proper local scoring rules," The Annals of Statistics,vol.40,no.1, Feb.2012.   
[39]F.Lin,X.Fang,and Z. Gao,“Distributionally robust optimization: A review on theory and applications,”Numerical Algebra, Control and Optimization,vol.12,no.1, pp.159-212,2022.   
[40] G.N. Iyengar,“Robust dynamic programming,” Mathematics of Operations Research,vol.30,no.2,pp.257-280,May 2005.   
[41]P.M. Pardalos and G. Schnitger,“Checking local optimality in constrained quadratic programming is np-hard,” Operations Research Letters,vol.7,no.1,pp.33-35,Feb.1988.   
[42] C.Jin,P.Netrapalli,and M. Jordan,“What is local optimality in nonconvex-nonconcave minimax optimization?’ in Proceedings of the 37th International Conference on Machine Learning.PMLR, Nov. 2020,pp.4880-4889.   
[43] C.Daskalakis,S. Skoulakis,and M. Zampetakis,“The complexity of constrained min-max optimization, in Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing,ser. STOC 2021. New York,NY,USA:Association for Computing Machinery,Jun.2021, pp. 1466-1478.   
[44] B.Grimmer,H. Lu,P.Worah,and V.Mirrokni,“Limiting behaviors of nonconvex-nonconcave minimax optimization via continuous-time systems,’in Proceedingsof The 33rd International Conferenceon Algorithmic Learning Theory． PMLR,Mar.2022,pp. 465-487.   
[45]T.Fiez,B.CasoadLRati,“licit eangdacs elberggames:Equilibria characterization,convergenceanalysis,and empirical study’in Proceedings of the 37th International Conference on Machine Learning． PMLR,Nov.2020,pp. 3133-3144.   
[46]K.M.LynchandF.C.Park,Modern Robotics: Mechanics,Planning, and Control,1st ed.Cambridge: Cambridge University Press,Jul.2017.

# APPENDIXA PROOF OF THEOREM1

Proof. For $k \in \{ 0 , \ldots , T - 1 \}$ and any $z \in { \mathcal { Z } }$ ，we denote $\begin{array} { r } { W _ { k } ( z ) = \operatorname* { s u p } _ { \hat { p } _ { k } \in \mathcal { F } } \operatorname* { i n f } _ { \rho _ { k } \in \mathbb { Z } _ { k } ( z ) } \int _ { \mathcal { X } } \mathcal { L } ( \hat { p } , x ) + V _ { k + 1 } ^ { * } ( z ^ { \prime } ) \mathrm { d } \rho _ { k } ( x ) } \end{array}$ Step I: $V _ { k } ^ { * } ( z ) \overset { } { \ } \leq W _ { k } ( z )$ .According to the definitions (4),(5), and (6), we have

$$
\begin{array} { r l } { V _ { \xi ^ { * } } ^ { * } ( z ) = \underset { \mathcal { F } _ { \xi ^ { * } + 1 } , \ldots , \mathcal { F } _ { \xi ^ { * } + 1 } } { \operatorname* { s u p } } \Bigg [ \frac { \mathrm { i n } \{ \sum _ { j \in \mathcal { F } _ { i } } \sum _ { \ell = 1 } ^ { i } \bigg [ \displaystyle \sum _ { \ell = 1 } ^ { i - 1 } \sum _ { \ell = 1 } ^ { i } \big ( \mathcal { F } _ { i } ( z ) , x _ { \ell + 1 } ) \big ] \bigg \{ z _ { k } = z \bigg \} } } { \displaystyle \sum _ { \ell = 1 } ^ { i } \operatorname* { s u p } _ { z \in \mathcal { F } _ { i } } \Big [ \displaystyle \sum _ { \ell = 1 } ^ { i - 1 } \sum _ { \ell = 1 } ^ { i } \Big ( \mathcal { F } _ { i } ( z ) , x _ { \ell + 1 } ) \Big ] + }  & { } \\ { \overset { ( \mathrm { i n } ) \leq \mathrm { i n } \{ \displaystyle \sum _ { \ell \in \mathcal { F } _ { i } } \sum _ { \ell = 1 } ^ { i } \sum _ { \ell = 1 } ^ { i } \sum _ { \ell = 1 } ^ { i } \Big [ \xi \big ( \mathcal { F } _ { i } ( z ) , x _ { \ell + 1 } ) + \xi \big ] } } { \displaystyle \sum _ { \ell \in \mathcal { F } _ { i } + 1 - 1 } ^ { i - 1 } \sum _ { \ell = 1 } ^ { i } \sum _ { \ell = 1 } ^ { i } \big [ \xi \big ( \mathcal { F } _ { i } ( z ) , x _ { \ell + 1 } ) \big ] \Big \} \Bigg ] z _ { k } = z \Bigg \} } & { } \\  \overset { ( \mathrm { i n } ) } { = } \underset { \mathcal { F } _ { \xi ^ { * } + 1 } , \ldots , \{ \xi \} } { \operatorname* { s u p } } \Bigg [ \mathrm { i n } \{ \displaystyle \sum _ { \ell = 1 } ^ { i } \xi \big ( \mathcal { F } _ { i } ( z ) , x _ { \ell + 1 } ) - \xi \big ( \mathcal { F } _ { i } ( z ) , x _ { \ell + 1 } \ \end{array}
$$

where $( i )$ holds because $\mathcal { P } _ { k + 1 : T - 1 }$ do not affect the first term $\mathcal { L } ( \mathcal { F } _ { k } ( z ) , \pmb { x } _ { k + 1 } )$ . According to Definition 1 and 2,it follows that for all $\mathcal { F } \in \mathfrak { F }$ $\mathcal { P } \in \mathfrak { P }$ ， $z \in { \mathcal { Z } }$ ，and $t \in \{ k , \ldots , T - 1 \}$ ， there is

$$
\begin{array} { r l } & { \quad \mathbb { E } _ { \mathcal { P } _ { t } } \vert \mathcal { L } ( \mathcal { F } _ { t } ( z _ { t } ) , \pmb { x } _ { t + 1 } ) \vert z _ { t } = z \vert } \\ & { \le \mathbb { E } _ { \mathcal { P } _ { t } } \left[ C ( 1 + \Vert \pmb { x } _ { t + 1 } \Vert _ { 2 } ^ { 2 } ) \vert z _ { t } = z \right] < \infty . } \end{array}
$$

Therefore, $( i i )$ interchanges $\operatorname { i n f } _ { \mathcal { P } _ { k + 1 : T - 1 } }$ and $\mathbb { E } _ { \mathcal { P } _ { k } }$ according to the dominated convergence theorem.

Votice that $V _ { k + 1 } ^ { \mathcal { F } } ( z ) \leq \bar { V } _ { k + 1 } ^ { \ast } ( z ) , \forall z \in \mathcal { Z }$ ， thus one has

$$
\begin{array} { r l } & { V _ { k } ^ { * } ( z ) \leq \underset { \mathcal { F } _ { k } } { \operatorname* { s u p } } \underset { \mathcal { P } _ { k } } { \operatorname* { i n f } } \mathbb { E } _ { \mathcal { P } _ { k } } \biggr [ \mathcal { L } ( \mathcal { F } _ { k } ( z ) , \pmb { x } _ { k + 1 } ) + V _ { k + 1 } ^ { * } ( z _ { k + 1 } ) \biggr | z _ { k } = z \biggr ] } \\ & { \qquad \overset { ( i ) } { = } \underset { \hat { p } _ { k } \in \mathcal { F } ^ { \rho _ { k } } } { \operatorname* { s u p } } \underset { \in \mathcal { T } _ { k } \in \mathcal { T } _ { k } ( z ) } { \operatorname* { i n f } } \int _ { \mathcal { X } } \mathcal { L } ( \hat { p } , x ) + V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \mathrm { d } \rho _ { k } ( x ) } \\ & { \qquad = W _ { k } ( z ) , } \end{array}
$$

where $( i )$ holds because: first, $\mathcal { F } _ { k }$ affect the first term by $\mathcal { F } _ { k } ( z ) \in \mathcal { F }$ and does notaffet the second tem $V _ { k + 1 } ^ { * } ( z _ { k + 1 } )$ second, given $z _ { k } ~ = ~ z$ ，one has the conditional measure $P _ { k } ( \cdot \mid z ) \in \mathcal { T } _ { k } ( z )$ according to Assumption 3.

Step II: $V _ { k } ^ { * } ( z ) \geq W _ { k } ( z )$ ．Accordingtothedefinition $\begin{array} { r } { \overline { { V _ { k + 1 } ^ { * } ( z ) } } = \operatorname* { s u p } _ { \mathcal { F } \in \mathfrak { F } } V _ { k + 1 } ^ { \mathcal { F } } ( z ) } \end{array}$ i follows that for all $\epsilon > 0$ ,there exists a predictor $\mathcal { F } ^ { \epsilon } \in \mathfrak { F }$ such that $V _ { k + 1 } ^ { \mathcal { F } ^ { \epsilon } } ( z ) \geq V _ { k + 1 } ^ { * } ( z ) - \epsilon$ for all $z \in { \mathcal { Z } }$ . Therefore,

$$
\begin{array} { r l } & { V _ { k } ^ { * } ( z ) = \underset { \mathscr { F } _ { k : T - 1 } } { \operatorname* { s u p } } \underset { \mathscr { P } _ { k } } { \operatorname* { i n f } } \mathbb { E } _ { \mathcal { P } _ { k } } \bigg [ \mathcal { L } ( \mathcal { F } _ { k } ( z ) , \boldsymbol { x } _ { k + 1 } ) + V _ { k + 1 } ^ { \mathcal { F } } ( \boldsymbol { z } _ { k + 1 } ) \bigg | \boldsymbol { z } _ { k } = \boldsymbol { z } \bigg ] } \\ & { \qquad \geq \underset { \mathscr { F } _ { k } } { \operatorname* { s u p } } \underset { \mathscr { P } _ { k } } { \operatorname* { i n f } } \mathbb { E } _ { \mathcal { P } _ { k } } \bigg [ \mathcal { L } ( \mathcal { F } _ { k } ( z ) , \boldsymbol { x } _ { k + 1 } ) + V _ { k + 1 } ^ { \mathcal { F } ^ { \epsilon } } ( \boldsymbol { z } _ { k + 1 } ) \bigg | \boldsymbol { z } _ { k } = \boldsymbol { z } \bigg ] } \\ & { \qquad \geq \underset { \mathscr { F } _ { k } } { \operatorname* { s u p } } \underset { \mathscr { P } _ { k } } { \operatorname* { i n f } } \mathbb { E } _ { \mathcal { P } _ { k } } \bigg [ \mathcal { L } ( \mathcal { F } _ { k } ( z ) , \boldsymbol { x } _ { k + 1 } ) + V _ { k + 1 } ^ { * } ( \boldsymbol { z } _ { k + 1 } ) \bigg | \boldsymbol { z } _ { k } = \boldsymbol { z } \bigg ] } \\ & { \qquad - \epsilon } \end{array}
$$

Since $\epsilon > 0$ is arbitrary, it implies that $V _ { k } ^ { * } ( z ) \geq W _ { k } ( z )$ Combining the results from step 1 and step 2，we have $V _ { k } ^ { * } ( z ) = W _ { k } ( z )$ for all $z \in { \mathcal { Z } }$ □

# APPENDIX B DUALITY OF PROBLEM (8)

Proof. The Lagrange function for (8) is defined as

$$
\begin{array} { r l } & { \quad L ( \rho _ { k } , \nu _ { k } , \mu _ { k } , r , q , Q _ { 1 } , Q _ { 2 } , P _ { 1 } , p _ { 1 } , s _ { 1 } , P _ { 2 } , p _ { 2 } , s _ { 2 } ) } \\ & { : = \langle \rho _ { k } , \log \hat { p } _ { k } \rangle + r \left[ \langle \rho _ { k } , 1 \rangle - 1 \right] + q ^ { \top } \lbrack \langle \rho _ { k } , x \rangle - \mu _ { k } - \nu _ { k } \rbrack } \\ & { \quad + Q _ { 1 } \cdot \left[ \langle \rho _ { k } , ( x - \nu _ { k } - \bar { \mu } _ { k } ) ( x - \nu _ { k } - \bar { \mu } _ { k } ) ^ { \top } \rangle - \gamma _ { 2 } \bar { \Sigma } _ { k } \right] } \\ & { \quad - Q _ { 2 } \cdot \left[ \langle \rho _ { k } , ( x - \nu _ { k } - \bar { \mu } _ { k } ) ( x - \nu _ { k } - \bar { \mu } _ { k } ) ^ { \top } \rangle - \gamma _ { 3 } \bar { \Sigma } _ { k } \right] } \\ & { \quad - P _ { 1 } \cdot \bar { \Sigma } _ { k } - 2 p _ { 1 } ^ { \top } ( \mu - \bar { \mu } _ { k } ) - s _ { 1 } \gamma _ { 1 } } \\ & { \quad - P _ { 2 } \cdot I - 2 p _ { 2 } ^ { \top } ( \nu _ { k } - \bar { \nu } _ { k } ) - s _ { 2 } \gamma _ { 0 } ( z ) . } \end{array}
$$

where the Lagrange multipliers satisfy that $Q _ { i } ~ \succeq ~ 0$ and $\left\lceil \begin{array} { l l } { P _ { i } } & { p _ { i } } \\ { p _ { i } ^ { \top } } & { s _ { i } } \end{array} \right\rceil \succeq 0$ for $i \in \{ 1 , 2 \}$ . Minimizing over $\rho _ { k } \in \mathcal { M } _ { + } ( \mathcal { X } )$ ， $\bar { \mu } _ { k } \in \mathbb { R } ^ { d _ { x } ^ { - } }$ ,and $\nu _ { k } \in \mathbb { R } ^ { d _ { x } }$ we have

$$
\begin{array} { r l } & { \quad _ { p } \frac { 1 } { \mu _ { 0 } \mu _ { 1 } } \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } } \\ & { = \int _ { \mathbb { R } ^ { 3 } } \frac { 1 } { \mu _ { 0 } \mu _ { 1 } \mu _ { 1 } } \int _ { \mathbb { R } ^ { 3 } } \int _ { \mathbb { R } ^ { 3 } } \left( \sigma \right) \cdot _ { 1 } \ \tau \ \mathfrak { L } _ { 1 } \ \ \eta _ { 2 } \ \left| \zeta _ { 1 } \ \ \eta _ { 3 } \right| ^ { 2 } \ \ \zeta _ { 2 } | \ \ \mathfrak { L } _ { 1 } \ \ \big ( \mathbf { Q } _ { 1 } - \mathbf { Q } _ { 2 } \big ) | } \\ &  \quad _ { p } \frac { 1 } { \mu _ { 0 } \mu _ { 1 } \mu _ { 1 } } \ \cdot \ \ \mathbf { U } _ { 1 } ^ { \prime } \big ( \mathbf { L } _ { 1 } ^ { \prime } \big ( \mathbf { Q } _ { 1 } - \mathbf { U } _ { 1 } ^ { \prime } \big ( \mathbf { Q } _ { 1 } ^ { \prime } \big ) \big ) ^ { \frac { 1 } { \sqrt { 3 } } } \end{array}
$$

where $q + 2 p _ { 1 } + 2 ( Q _ { 1 } - Q _ { 2 } ) ( \nu _ { k } + \bar { \mu } _ { k } ) = 0$ Equation $( i )$ holds when $\log \hat { p } _ { k } ( x ) + r + q ^ { \top } x + x ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) x +$ $V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) ~ \geq ~ 0 ~ \forall x ~ \in ~ \mathbb { R } ^ { d _ { x } }$ ，otherwise there is ${ \mathrm { i n f } } _ { \rho _ { k } \in { \mathcal { M } } _ { + } ( { \mathcal { X } } ) } L$ equals $- \infty$ . Equation $( i i )$ holds because

$$
\begin{array} { r l } & { \quad \underset { \mu _ { k } , \nu _ { k } } { \mathrm { i n f } } - ( q + 2 p _ { 1 } ) ^ { \top } \mu _ { k } - ( q + 2 p _ { 2 } ) ^ { \top } \nu _ { k } - \bar { \mu } _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \bar { \mu } _ { k } } \\ & { \quad - ( \nu _ { k } + 2 \mu _ { k } - \bar { \mu } _ { k } ) ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) ( \nu _ { k } + \bar { \mu } _ { k } ) } \\ & { = \underset { \nu _ { k } } { \mathrm { i n f } } \underset { \mu _ { k } } { \mathrm { i n f } } - [ q + 2 p _ { 1 } + 2 ( Q _ { 1 } - Q _ { 2 } ) ( \nu _ { k } + \bar { \mu } _ { k } ) ] ^ { \top } \mu _ { k } } \\ & { \quad \quad - ( q + 2 p _ { 2 } ) ^ { \top } \nu _ { k } - \nu _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \nu _ { k } } \\ & { = \underset { \nu _ { k } } { \mathrm { i n f } } - ( q + 2 p _ { 2 } ) ^ { \top } \nu _ { k } - \nu _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \nu _ { k } } \\ & { \quad \quad \mathrm { s . t . } \ q + 2 p _ { 1 } + 2 ( Q _ { 1 } - Q _ { 2 } ) ( \nu _ { k } + \bar { \mu } _ { k } ) = 0 . } \end{array}
$$

Combining those conditions and the previous multipliers' constraints,we have the dual form completed. □

# APPENDIX C

# PROOF OF THEOREM 2

Proof. Step I. We leverage the separable structure of $\mathbf { D } _ { 1 }$ to reformulate the problem. For ease of notation,we use the $\kappa$ to summarize all the Lagrange multipliers except for $r$ ,i.e.,

$$
\kappa = ( q , Q _ { 1 } , Q _ { 2 } , P _ { 1 } , p _ { 1 } , s _ { 1 } , P _ { 2 } , p _ { 2 } , s _ { 2 } ) .
$$

First, the objective function can be separated as $G ( \boldsymbol { r } , \kappa ) =$ $- r + g ( \kappa )$ where $g ( \kappa ) = { \bar { \mu } _ { k } } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) { \bar { \mu } _ { k } } - ( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } \dot { Q } _ { 2 } +$ $P _ { 1 } ) \cdot \bar { \Sigma } _ { k } - P _ { 2 } \cdot I + 2 p _ { 1 } ^ { \top } \bar { \mu } _ { k } - s _ { 1 } \gamma _ { 1 } + 2 p _ { 2 } ^ { \top } \bar { \nu } _ { k } - s _ { 2 } \gamma _ { 0 } ( z ) ^ { 2 } -$ $( q + 2 p _ { 2 } ) ^ { \top } \nu _ { k } ^ { * } - \nu _ { k } ^ { * , \top } ( Q _ { 1 } - Q _ { 2 } ) \nu _ { k } ^ { * }$ .Second, only one of the constraints concerns $r$ . Therefore, given any group of feasible $\kappa$ satisfying the other constraints in $\mathbf { D } _ { 1 }$ ， the optimizing over $\hat { p } _ { k }$ and $r$ is equivalent to solving the following problem:

$$
\begin{array} { r l } & { \underset { \hat { p } _ { k } \in \mathcal { F } , r } { \operatorname* { s u p } } - r + g ( \kappa ) } \\ & { \mathrm { s . t . } \left\{ x ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) x + x ^ { \top } q + r + \log \hat { p } _ { k } ( x ) \right. } \\ & { \left. \vphantom { \int _ { 0 } ^ { 0 } } + V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \geq 0 \forall x \in \mathcal { X } \right. } \end{array}
$$

Step II. We exploit the last constraint in (2O) to analyze the lower bound of $r$ . Notice that

$$
\begin{array} { r l } & { 1 = \displaystyle \int _ { \mathbb R ^ { d _ { x } } } \hat { p } _ { k } ( x ) \mathrm { d } x } \\ & { ~ \geq \displaystyle \int _ { \mathbb R ^ { d _ { x } } } \exp \big \{ - x ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) x - x ^ { \top } q - r } \\ & { ~ - V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \big \} \mathrm { d } x , } \end{array}
$$

one immediately gets that $r$ is lower bounded by $r ^ { * } : =$ $\begin{array} { r } { \log [ \int _ { \mathbb { R } ^ { d _ { x } } } \exp \{ - x ^ { \top } ( \bar { Q } _ { 1 } - Q _ { 2 } ) x - x ^ { \top } q - V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \} \mathrm { d } x ] . } \end{array}$

Notice that the equality of (21) holds only when there is $\hat { p } _ { k } ( x ) = \exp \{ - x ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) x - x ^ { \top } q - r ^ { * } \}$ for $x \in \mathbb { R } ^ { d _ { x } }$ almost everywhere.Therefore，given any feasible $\kappa$ ，the objective function $- r + g ( \kappa )$ in problem(20） can only attain its maximum when $\hat { p } _ { k }$ belongs to the following exponential family almost everywhere on $\mathbb { R } ^ { d _ { x } }$ ，

$$
\begin{array} { r } { \hat { p } _ { k } ( x ) \overset { a . s . } { \propto } \exp \{ - x ^ { \top } \theta _ { 1 } x - x ^ { \top } \theta _ { 2 } - V _ { k + 1 } ^ { * } ( x , \pi _ { k + 1 } ( x ) ) \} , } \end{array}
$$

where $\theta _ { 1 } \in S ^ { d _ { x } }$ and $\theta _ { 2 } \in \mathbb { R } ^ { d _ { x } }$ .

Step I. Finally, when $k = T - 1$ ,there is $V _ { T } ^ { * } ( x , \pi _ { T } ( x ) ) =$ 0. Because $Q _ { 1 }$ and $Q _ { 2 }$ are both semi-definite,we have $Q _ { 1 } -$ $Q _ { 2 }$ is symmetric and there exists an orthogonal matrix $U \in$ $\mathbb { R } ^ { d _ { x } \times d _ { x } }$ and an diagonal matrix $D = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d _ { x } } )$ such that $Q _ { 1 } - Q _ { 2 } = U D U ^ { \top }$ . Then we have

$$
\begin{array} { r l } & { \displaystyle \int _ { \mathbb R ^ { d _ { x } } } \exp \{ - x ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) x - x ^ { \top } q - r \} \mathrm d x } \\ & { = \displaystyle \int _ { \mathbb R ^ { d _ { x } } } \exp \{ - x ^ { \top } U D U ^ { \top } x - x ^ { \top } q - r \} \mathrm d x } \\ & { = \displaystyle \int _ { \mathbb R ^ { d _ { x } } } \exp \{ - \tilde { x } ^ { \top } D \tilde { x } - \tilde { x } ^ { \top } \tilde { q } - r \} \operatorname* { d e t } ( U ) \mathrm d \tilde { x } } \\ & { = e ^ { - r } \operatorname* { d e t } ( U ) \displaystyle \prod _ { i = 1 } ^ { d _ { x } } \int _ { \mathbb R } \exp \{ - \tilde { x } _ { ( i ) } ^ { 2 } \lambda _ { i } - \tilde { x } _ { ( i ) } \tilde { q } _ { ( i ) } \} \mathrm d \tilde { x } _ { ( i ) } , } \end{array}
$$

where the second equation follows by letting $\tilde { x } = U ^ { \top } x$ and $\tilde { q } \ = \ U ^ { \top } q$ . If there is an index $j ~ \in ~ \{ 1 , \dots , d _ { x } \}$ such that $\lambda _ { j } < 0$ ，then $\begin{array} { r } { \int _ { \mathbb { R } } \exp \{ - \tilde { x } _ { ( i ) } ^ { 2 } \lambda _ { j } - \tilde { x } _ { ( j ) } \tilde { q } _ { ( j ) } \} \mathrm { d } \tilde { x } _ { ( j ) } = \infty } \end{array}$ ， which contradicts (21). Therefore, $Q _ { 1 } - Q _ { 2 }$ is semi-definite positive, and a necessary optimal condition for the inner optimization is that $\hat { p } _ { T - 1 }$ is subject to Gaussian almost everywhere. □

# APPENDIX D DERIVATION OF（11)

Proof. Facilitated by Theorem 2,when $k = T - 1$ we can first parameterize $\hat { p } _ { k }$ by $\hat { \mu } _ { k } \in \mathbb { R } ^ { d _ { x } }$ and $\hat { \Sigma } _ { k } \in S _ { + } ^ { d _ { x } }$ such that

$$
\hat { p } _ { k } ( x ) = \frac { 1 } { \sqrt { ( 2 \pi ) ^ { d _ { x } } \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) } } \exp \left[ - \frac { 1 } { 2 } ( x - \hat { \mu } _ { k } ) ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } ( x - \hat { \mu } _ { k } ) \right] .
$$

Substituting it into the constraints of $\mathbf { D } _ { 1 }$ , there is

$$
\begin{array} { r l } & { x ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) x + x ^ { \top } q + r - \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) \right] } \\ & { - \frac { 1 } { 2 } ( x - \hat { \mu } _ { k } ) ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } ( x - \hat { \mu } _ { k } ) = 0 \quad \forall x \in \mathbb { R } ^ { d _ { x } } . } \end{array}
$$

Then we have $Q _ { 1 } \ - \ Q _ { 2 } \ = \ \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 }$ $q ~ = ~ - \hat { \Sigma } _ { k } ^ { - 1 } \hat { \mu } _ { k }$ and $\begin{array} { r } { r = \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) + \hat { \mu } _ { k } ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } \hat { \mu } _ { k } \right] } \end{array}$ .Substuting these equations and the parameterized $\hat { p } _ { k }$ into the objective of problem $\mathbf { D } _ { 1 }$ ，we get

$$
\begin{array} { r l } & { \quad G ( r , q , Q _ { 1 } , Q _ { 2 } , P _ { 1 } , p _ { 1 } , s _ { 1 } , P _ { 2 } , p _ { 2 } , s _ { 2 } ) } \\ & { = - \cfrac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \widehat { \Sigma } _ { k } ) \right] } \\ & { \quad - P _ { 2 } \cdot I - s _ { 1 } \gamma _ { 1 } + 2 p _ { 2 } ^ { \top } ( \bar { \nu } _ { k } + \bar { \mu } _ { k } - \hat { \mu } _ { k } ) - s _ { 2 } \gamma _ { 0 } ( z ) ^ { 2 } } \\ & { \quad + 2 ( 2 p _ { 2 } - p _ { 1 } ) ^ { \top } \widehat { \Sigma } _ { k } p _ { 1 } - ( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P _ { 1 } ) \cdot \widehat { \Sigma } _ { k } } \\ & { = - \cfrac { 1 } { 2 } d _ { x } \log ( 2 \pi ) + \cfrac { 1 } { 2 } \log \operatorname* { d e t } ( \widehat { \Sigma } _ { k } ^ { - 1 } ) - ( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } ) \cdot \bar { \Sigma } _ { k } } \\ & { \quad - P _ { 1 } \cdot \overline { { \Sigma } } _ { k } - P _ { 2 } \cdot I - s _ { 1 } \gamma _ { 1 } - s _ { 2 } \gamma _ { 0 } ( z ) ^ { 2 } } \\ & { \quad - 2 p _ { 2 } ^ { \top } ( \hat { \mu } _ { k } - \bar { \mu } _ { k } - \bar { \nu } _ { k } ) + 2 p _ { 1 } ^ { \top } \widehat { \Sigma } _ { k } ( 2 p _ { 2 } - p _ { 1 } ) . } \end{array}
$$

Substituting the above equations into the problem $\mathbf { D } _ { 1 }$ , we have the problem（11） derived. □

# APPENDIXE PROOF OFLEMMA 1

Proof. Since the objective function is linear with respect to $\Sigma _ { k }$ and the second constraint provides an upper bound for it, one immediately has the worst-case covariance should satisfy that

$$
\Sigma _ { k } ^ { * } = \gamma _ { 2 } \bar { \Sigma } _ { k } - ( \mu _ { k } - \bar { \mu } _ { k } ) ( \mu _ { k } - \bar { \mu } _ { k } ) ^ { \top } .
$$

Let $a = \mu _ { k } - { \bar { \mu } } _ { k } , b = \nu _ { k } - { \bar { \nu } } _ { k } , c = { \bar { \mu } } _ { k } + { \bar { \nu } } _ { k } - { \hat { \mu } } _ { k }$ ，problem $\mathbf { P } _ { 2 }$ can be further equivalent to

$$
\begin{array} { r l } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathrm { i n f } \quad \operatorname* { s u p } - \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ^ { - 1 } ) + \gamma _ { 2 } \bar { \Sigma } _ { k } { \cdot } \hat { \Sigma } _ { k } ^ { - 1 } } \\ & { \quad \quad \quad \quad \hat { \Sigma } _ { k } ^ { - 1 } , c \ a , b } \\ & { \quad \quad \quad \quad \quad \quad + \left\| a + b + c \right\| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \left\| a \right\| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } } \\ & { \quad \quad \quad \quad \mathrm { s . t . } \ \| a \| _ { \bar { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \leq \gamma _ { 1 } , \| b \| _ { 2 } ^ { 2 } \leq \gamma _ { 0 } ( z ) . } \end{array}
$$

Suppose $\begin{array} { r } { ( a ^ { * } , b ^ { * } ) \in \arg \operatorname* { m a x } _ { a , b } \left\{ \| a + b \| _ { \hat { \Sigma } ^ { - 1 } } ^ { 2 } - \| a \| _ { \hat { \Sigma } ^ { - 1 } } ^ { 2 } \right\} } \end{array}$ we have $( - a ^ { * } , - b ^ { * } )$ can also attain the maximum. Then we have

$$
\begin{array} { r l } & { \quad \underset { a , b } { \mathrm { m a x } } \left\| a + b + c \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \left\| a \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } } \\ & { \geq \operatorname* { m a x } \left\{ \left\| a ^ { * } + b ^ { * } + c \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \left\| a ^ { * } \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } , } \\ & { \qquad \left\| - a ^ { * } - b ^ { * } + c \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \left\| - a ^ { * } \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \right\} } \\ & { = \left\| a ^ { * } + b ^ { * } \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \left\| a ^ { * } \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } + \left\| c \right\| _ { \widehat { \Sigma } _ { k } } ^ { - 1 } } \\ & { \quad + 2 \operatorname* { m a x } \left\{ \left. c , a ^ { * } + b ^ { * } \right. _ { \widehat { \Sigma } _ { k } ^ { - 1 } } , - \left. c , a ^ { * } + b ^ { * } \right. _ { \widehat { \Sigma } _ { k } ^ { - 1 } } \right\} } \\ & { \geq \left\| a ^ { * } + b ^ { * } \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - \left\| a ^ { * } \right\| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } } \end{array}
$$

，

# APPENDIX F DUALITY OF THE MOMENT PROBLEM(15)

Proof. The Lagrange function for (15) is defined as

$$
\begin{array} { r l } & { \quad L ( \tilde { \rho } _ { k } , \mu _ { k } , r , q , Q _ { 1 } , Q _ { 2 } , P , p , s ) } \\ & { : = \langle \tilde { \rho } _ { k } , \log \hat { p } _ { k } ( w + \bar { \nu } _ { k } ) \rangle + r \left[ \langle \tilde { \rho } _ { k } , 1 \rangle - 1 \right] + q ^ { \top } \left[ \langle \tilde { \rho } _ { k } , w \rangle - \mu _ { k } \right] } \\ & { \quad + Q _ { 1 } \cdot \left[ \langle \tilde { \rho } _ { k } , ( w - \bar { \mu } _ { k } ) ( w - \bar { \mu } _ { k } ) ^ { \top } \rangle - \gamma _ { 2 } \bar { \Sigma } _ { k } \right] } \\ & { \quad - Q _ { 2 } \cdot \left[ \langle \tilde { \rho } _ { k } , ( w - \bar { \mu } _ { k } ) ( w - \bar { \mu } _ { k } ) ^ { \top } \rangle - \gamma _ { 3 } \bar { \Sigma } _ { k } \right] } \\ & { \quad - P \cdot \bar { \Sigma } _ { k } - 2 p ^ { \top } ( \mu _ { k } - \bar { \mu } _ { k } ) - s \gamma _ { 1 } , } \end{array}
$$

where the Lagrange multipliers satisfy $Q _ { 1 } \succeq 0 , Q _ { 2 } \succeq 0$ and $\left[ \begin{array} { l l } { P } & { p } \\ { p ^ { \top } } & { s } \end{array} \right] \succeq 0$ . Minimizing over $\tilde { \rho } _ { k } \in \mathcal { M } _ { + } ( \mathcal { X } )$ and $\mu _ { k } \in \mathbb { R } ^ { d _ { x } }$ ， one has

$$
\begin{array} { r l } & { \quad \underset { \tilde { \rho } _ { k } \in \mathcal { M } _ { + } ( \mathcal { X } ) , \mu _ { k } } { \operatorname* { i n f } } L ( \tilde { \rho } _ { k } , \mu _ { k } , r , q , Q _ { 1 } , Q _ { 2 } , P , p , s ) } \\ & { = \underset { \tilde { \rho } _ { k } \in \mathcal { M } _ { + } ( \mathcal { X } ) , \mu _ { k } } { \operatorname* { i n f } } \langle \tilde { \rho } _ { k } , \log \hat { p } _ { k } ( w + \bar { \nu } _ { k } ) + r + q ^ { \top } w + w ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) w \rangle } \\ & { \quad - \left[ q + 2 ( Q _ { 1 } - Q _ { 2 } ) \hat { \mu } _ { k } + 2 p \right] ^ { \top } \mu _ { k } - r + \hat { u } _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \hat { u } _ { k } } \\ & { \quad - \left( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P \right) \cdot \hat { \sum } _ { k } + 2 p ^ { \top } \hat { \mu } _ { k } - s \gamma _ { 1 } } \\ & { = - r + \hat { u } _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \hat { u } _ { k } - ( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P ) \cdot \hat { \sum } _ { k } + 2 p ^ { \top } \hat { \mu } _ { k } } \\ & { \quad - s \gamma _ { 1 } , } \end{array}
$$

where the second equality holds when $\log \hat { p } _ { k } ( w + \bar { \nu } _ { k } ) + r +$ $q ^ { \top } w + w ^ { \top } ( Q _ { 1 } - \bar { Q } _ { 2 } ) w \ \geq \ 0 \ \forall w \ \in \ \mathbb { R } ^ { d _ { x } }$ and $q + 2 ( Q _ { 1 } -$ $Q _ { 2 } ) \bar { \mu } _ { k } + 2 p = 0$ hold simultaneously,otherwise there is $\begin{array} { r } { \operatorname* { i n f } _ { { \tilde { \rho } } _ { k } \in { \mathcal { M } } _ { + } ( { \mathcal { X } } ) , \mu _ { k } } L = - \infty } \end{array}$ ：Combining these two conditions and the previous multipliers’constraints,we have attained the dual problem $\mathbf { D } _ { 2 }$ □

# APPENDIX G PROOF OF THEOREM 3

Proof. The proof is organized in three parts: first, we use Theorem 2 to parameterize $\hat { p } _ { k }$ by Gaussian distribution $\mathcal { N } ( \hat { \mu } _ { k } , \hat { \Sigma } _ { k } )$ and reformulate the dual problem $\mathbf { D } _ { 2 }$ to a finite-dimensional optimization; second,we solve the optimal predictive pdf $\hat { p } _ { k } ^ { * } \sim \mathcal N ( \hat { \mu } _ { k } ^ { * } , \hat { \Sigma } _ { k } ^ { * } )$ ； third, we substitute $\hat { p } _ { k } ^ { * }$ into the original problem ${ \bf P } _ { 1 }$ to obtain the worst-case conditional measure $\rho _ { k } ^ { * }$ by solving the inner minimization.

Step I. Using the necessary optimality condition for $\hat { p } _ { k }$ ，we can parameterize $\hat { p } _ { k }$ by $\hat { \mu } _ { k } \in \mathbb { R } ^ { d _ { x } }$ and $\hat { \Sigma } _ { k } \in S _ { + } ^ { d _ { x } }$ such that

$$
\hat { p } _ { k } ( w + \bar { \nu } _ { k } ) = \frac { \exp \left[ - \frac { 1 } { 2 } ( w - \hat { \mu } _ { k } ) ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } ( w - \hat { \mu } _ { k } ) \right] } { \sqrt { ( 2 \pi ) ^ { d _ { x } } \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) } } .
$$

Moreover, from the constraint $w ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) w + w ^ { \top } q + r +$ $\log \hat { p } _ { k } ( w + \bar { \nu } _ { k } ) \geq 0 \forall w \in \mathbb { R } ^ { d _ { x } }$ ，we get

$$
\begin{array} { r l } & { w ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) w + w ^ { \top } q + r - \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) \right] } \\ & { - \frac { 1 } { 2 } ( w - \hat { \mu } _ { k } ) ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } ( w - \hat { \mu } _ { k } ) = 0 \quad \forall w \in \mathbb { R } ^ { d _ { x } } , } \end{array}
$$

whic indicates that $\begin{array} { r } { Q _ { 1 } - Q _ { 2 } = \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } } \end{array}$ $\boldsymbol { q } = - \hat { \Sigma } _ { k } ^ { - 1 } \hat { \mu } _ { k }$ ，and $\begin{array} { r } { r = \frac { 1 } { 2 } \lceil d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) \rceil + \tilde { \mu } _ { k } ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } \hat { \mu } _ { k } \rceil } \end{array}$ .Next,from the constraint $q + 2 ( Q _ { 1 } - Q _ { 2 } ) \bar { \mu } _ { k } + 2 p = 0$ one can get $p =$ $\begin{array} { r } { \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } ( \hat { \mu } _ { k } - \bar { \mu } _ { k } ) } \end{array}$ Substitutingtheparameterized $\hat { p } _ { k }$ and the above equations into problem $\mathbf { D } _ { 2 }$ ， there is

$$
\begin{array} { l } { { G ( r , q , Q _ { 1 } , Q _ { 2 } , P , p , s ) = - \displaystyle \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ) \right] } } \\ { { \phantom { G ( r , q , Q _ { 1 } , Q _ { 2 } , P , p , s ) = - \displaystyle \frac { 1 } { 2 } \hat { \mu } _ { k } ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } \hat { \mu } _ { k } + \bar { \mu } _ { k } ^ { \top } ( Q _ { 1 } - Q _ { 2 } ) \bar { \mu } _ { k } + \bar { \mu } _ { k } ^ { \top } \hat { \Sigma } _ { k } ^ { - 1 } ( \hat { \mu } _ { k } - \bar { \mu } _ { k } ) } } } \\ { { \phantom { G ( r , q , \hat { Q } _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P ) } \cdot \bar { \Sigma } _ { k } - s \gamma _ { 1 } } } \\ { { \phantom { G ( r , q , \hat { Q } _ { 1 } ) = - \displaystyle \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) - \frac { 1 } { 2 } \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ^ { - 1 } ) + \| \hat { \mu } _ { k } - \bar { \mu } _ { k } \| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \right] } } } \\ { { \phantom { G ( r , q , \hat { Q } _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P ) } \cdot \bar { \Sigma } _ { k } - s \gamma _ { 1 } . } } \end{array}
$$

As a result, $\mathbf { D } _ { 2 }$ can be transformed into a finite-dimensional optimization problem as follows:

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { \hat { a } , k , \hat { \Sigma } _ { k } , Q _ { 1 } , Q _ { 2 } , P , p , s } - \frac { 1 } { 2 } d _ { x } \log ( 2 \pi ) + \frac { 1 } { 2 } \log \operatorname* { d e t } ( \hat { \Sigma } _ { k } ^ { - 1 } ) - s \gamma _ { 1 } } \\ { \displaystyle - \frac { 1 } { 2 } \| \hat { \mu } _ { k } - \bar { \mu } _ { k } \| _ { \hat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } - ( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P ) \cdot \bar { \Sigma } _ { k } } \\ { \displaystyle \operatorname { s . t . } \left\{ \begin{array} { l l } { Q _ { 1 } \succeq 0 , Q _ { 2 } \succeq 0 , Q _ { 1 } - Q _ { 2 } = \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } , \hat { \Sigma } _ { k } \succ 0 } \\ { \bigg [ \begin{array} { l l } { P } & { p } \\ { p ^ { \top } } & { s } \end{array} \bigg ] \succeq 0 , p = \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } ( \hat { \mu } _ { k } - \bar { \mu } _ { k } ) . } \end{array} \right. } \end{array}
$$

Step II. Notice that the constraint $\hat { \Sigma } _ { k } \succ 0$ and the relationship $\begin{array} { r } { \overline { { p = \frac 1 2 \hat { \Sigma } _ { k } ^ { - 1 } ( \hat { \mu } _ { k } - \bar { \mu } _ { k } ) } } } \end{array}$ reveals that $p$ and $\hat { \mu } _ { k }$ are one-on-one, one can optimize on either of them.In this proof,we choose to maximize over $p$ and substitute $\hat { \mu } _ { k }$ in the objective by $p$ Next,notice that the constraints for $Q _ { 1 } , Q _ { 2 }$ only depend on $\hat { \Sigma } _ { k }$ ,and it can be separated from the constraints for $P , p , s$ we can first maximize over $Q _ { 1 }$ and $Q _ { 2 }$ ，then over $P , p , s$ ，finally over $\hat { \Sigma }$ .In other words,the objective of problem (23) can be reformulated as

$$
\begin{array} { l } { { \mathrm { s u p ~ s u p ~ } \displaystyle \operatorname* { s u p } _ { \hat { \Sigma } _ { k } ~ P , p , s _ { { \cal Q } } _ { 1 } , { \cal Q } _ { 2 } } \displaystyle - \frac { 1 } { 2 } d _ { x } \log ( 2 \pi ) + \displaystyle \frac { 1 } { 2 } \log \mathrm { d e t } ( \hat { \Sigma } ^ { - 1 } ) } } \\ { { \hat { \Sigma } _ { k } ~ P , p , s _ { { \cal Q } _ { 1 } , { \cal Q } _ { 2 } } } } \\ { { - ~ 2 p ^ { \top } \hat { \Sigma } p - ( \gamma _ { 2 } Q _ { 1 } - \gamma _ { 3 } Q _ { 2 } + P ) { \cdot } \bar { \Sigma } _ { k } - s \gamma _ { 1 } . } } \end{array}
$$

The solution of the first maximization is $\begin{array} { r } { Q _ { 1 } ^ { * } = \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } , Q _ { 2 } ^ { * } = } \end{array}$ O because

$$
\begin{array} { r l r } & { } & { Q _ { 1 } ^ { * } = \arg \underset { Q _ { 1 } \succeq \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } } { \mathrm { m a x } } - ( ( \gamma _ { 2 } - \gamma _ { 3 } ) Q _ { 1 } + \frac { \gamma _ { 3 } } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } ) \cdot \bar { \Sigma } _ { k } } \\ & { } & \\ & { } & { = \arg \underset { Q _ { 1 } \succeq \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } } { \mathrm { m a x } } - ( \gamma _ { 2 } - \gamma _ { 3 } ) Q _ { 1 } \cdot \bar { \Sigma } _ { k } = \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } , } \end{array}
$$

and $\begin{array} { r } { Q _ { 2 } ^ { * } = Q _ { 1 } ^ { * } - \frac { 1 } { 2 } \hat { \Sigma } _ { k } ^ { - 1 } = 0 } \end{array}$

The solution of the second maximization is $P ^ { * } = 0 , p ^ { * } =$ $0 , s ^ { * } ~ = ~ 0$ Notice that $\left[ \begin{array} { l l } { P } & { p } \\ { p ^ { \top } } & { s } \end{array} \right] ~ \succeq ~ 0$ is equivalent to $s ~ \geq ~ p ^ { \top } P ^ { - 1 } p$ ， therefore $s \geq 0$ and $\arg \operatorname* { m a x } _ { s } - s \gamma _ { 1 } \ = \ 0$ Again, $\arg \operatorname* { m a x } _ { p } - 2 p ^ { \top } \hat { \Sigma } _ { k } p = 0$ and a $\mathrm { \cdot g } \operatorname* { m a x } _ { P } P \cdot \bar { \Sigma } _ { k } = 0$ Additionally, $p ^ { * } = 0$ indicates that $\hat { \mu } _ { k } ^ { * } = \hat { 2 } \hat { \Sigma } _ { k } p + \bar { \mu } _ { k } = \bar { \mu } _ { k }$

The solution of the third maximization is $\hat { \Sigma } _ { k } ^ { * } = \gamma _ { 2 } \bar { \Sigma } _ { k }$ . This is because $\hat { \Sigma } _ { k } ^ { * } = [ \underset { - } { \arg \operatorname* { m a x } } _ { \hat { \Sigma } _ { k } ^ { - 1 } \succ 0 } F ( \hat { \Sigma } _ { k } ^ { - 1 } ) ] ^ { - 1 }$ where $F ( X ) =$ $\log \operatorname* { d e t } ( X ) - X \cdot \gamma _ { 2 } { \bar { \Sigma } } _ { k }$ . Notice that $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } X } F ( X ) = X ^ { - 1 } - \gamma _ { 2 } \bar { \Sigma } _ { k } } \end{array}$ and ${ \frac { \operatorname { d } } { \operatorname { d } X } } F$ monotonously decrese on $X$ Therefore, $F$ is concave, which can be maximized at $X ^ { * }$ where $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } X } F ( X ^ { * } ) = 0 } \end{array}$ i.e., $X ^ { * } = ( \gamma _ { 2 } \bar { \Sigma } _ { k } ) ^ { - 1 }$ . Finally, we have $\hat { \Sigma } _ { k } ^ { * }$ equals the inverse of $X ^ { * }$ ，which is $\gamma _ { 2 } \bar { \Sigma } _ { k }$ . Therefore, the optimal predictive pdf is $\hat { p } _ { k } ^ { * } \sim \mathcal { N } \left( \bar { \nu } _ { k } + \bar { \mu } _ { k } , \gamma _ { 2 } \bar { \Sigma } _ { k } \right)$

Step III. Substituting the optimal $\hat { p } _ { k } ^ { * }$ into problem ${ \bf P } _ { 1 }$ ，we have the inner minimization problem as

$$
\begin{array} { l } { \displaystyle \operatorname* { i n f } _ { \tilde { \rho } _ { k } , \mu _ { k } } \mathbb { E } _ { w \sim \tilde { \rho } _ { k } } - \frac { 1 } { 2 } \left[ d _ { x } \log ( 2 \pi ) + \log \operatorname* { d e t } ( \gamma _ { 2 } \bar { \Sigma } _ { k } ) \right. } \\ { \displaystyle \qquad + ( w - \bar { \mu } _ { k } ) ^ { \top } ( \gamma _ { 2 } \bar { \Sigma } _ { k } ) ^ { - 1 } ( w - \bar { \mu } _ { k } ) \big ] } \\ { \displaystyle \mathrm { s . t . } \left\{ \begin{array} { l l } { \mathbb { E } _ { w \sim \tilde { \rho } _ { k } } [ 1 ] = 1 , \mathbb { E } _ { w \sim \tilde { \rho } _ { k } } [ w ] = \mu _ { k } } \\ { \gamma _ { 3 } \bar { \Sigma } _ { k } \preceq \mathbb { E } _ { w \sim \tilde { \rho } _ { k } } \left[ ( w - \bar { \mu } _ { k } ) ( w - \bar { \mu } _ { k } ) ^ { \top } \right] \preceq \gamma _ { 2 } \bar { \Sigma } _ { k } } \\ { \left[ \begin{array} { c c c } { \bar { \Sigma } _ { k } } & { ( \mu _ { k } - \bar { \mu } _ { k } ) } \\ { ( \mu _ { k } - \bar { \mu } _ { k } ) ^ { \top } } & { \gamma _ { 1 } } \end{array} \right] \succeq 0 . } \end{array} \right. } \end{array}
$$

The objective isminimized when $\mathbb { E } \bigg [ \big ( \pmb { w } _ { k } - \overline { { \mu } } _ { k } \big ) \big ( \pmb { w } _ { k } - \overline { { \mu } } _ { k } \big ) ^ { \mathrm { T } } \bigg ]$ equals $\gamma _ { 2 } \bar { \Sigma } _ { k }$ ,and the proof is completed.

Tao $\mathbf { X } \mathbf { u }$ (S'22) received a B.S.degree in the School of Mathematical Sciences from Shanghai Jiao Tong University (SJTU),Shanghai,China.He is currently working toward the Ph.D.degree with the Department of Automation, SJTU. He is a member of Intelligent of Wireless Networking and Cooperative Control Group.His research interests include probabilistic prediction,distributionally robust optimization,dynamic games,and robotics.

Jianping He (SM'19) is currently an associate professor in the Department of Automation at Shanghai Jiao Tong University.He received the Ph.D.degree in control science and engineering from Zhejiang University,Hangzhou, China, in 2013,and had been a research fellow in the Department of Electrical and Computer Engineering at University of Victoria,Canada,from Dec.2013 to Mar. 2O17.His research interests mainly include the distributed learning, control and optimization,security and privacy in network systems.

Dr.He serves as an Associate Editor for IEEE Trans.Control of Network Systems,IEEE Trans.on Vehicular Technology，IEEE Open Journal of Vehicular Technology，and KSI Trans.Internet and Information Systems. He was also a Guest Editor of IEEE TAC,International Journal of Robust and Nonlinear Control,etc.He was the winner of Outstanding Thesis Award, Chinese Association of Automation,2O15.He received the best paper award from IEEE WCSP'17,the best conference paper award from IEEE PESGM'17,and was a finalist for the best student paper award from IEEE ICCA'17,and the finalist best conference paper award from IEEE VTC20- FALL.