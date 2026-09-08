https://people.sc.fsu.edu/~jburkardt/py_src/bdf2/bdf2.p#! /usr/bin/env python3
#
def bdf2 ( f, tspan, y0, n ):

#*****************************************************************************80
#
## bdf2() uses backward difference formula 2 + fsolve() to solve an ODE.
#
#  Discussion:
#
#    The first step is taken using a Runge-Kutta method of order 2.
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 May 2022
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    function handle f: evaluates the right hand side of the ODE.  
#
#    real tspan(2): the starting and ending times.
#
#    real y0(m): the initial conditions. 
#
#    integer n: the number of steps.
#
#  Output:
#
#    real t(n+1,1), y(n+1,m): the solution estimates.
#
  from scipy.optimize import fsolve
  import numpy as np

  if ( np.ndim ( y0 ) == 0 ):
    m = 1
  else:
    m = len ( y0 )

  t = np.linspace ( tspan[0], tspan[1], n + 1 )
  y = np.zeros ( [ n + 1, m ] )

  dt = ( tspan[1] - tspan[0] ) / float ( n )

  for i in range ( 0, n + 1 ):

    if ( i == 0 ):

      y[i,:] = y0.copy ( )

    elif ( i == 1 ):

      to = t[i-1]
      yo = y[i-1,:]
      th = t[i-1] + 0.5 * dt 
      yh = y[i-1,:] + 0.5 * dt * f ( t[i-1], y[i-1,:] )

      yh = fsolve ( backward_euler_residual, yh, args = ( f, to, yo, th ) )

      y[i,:] = 2.0 * yh - y[i-1,:]

    else:

      t1 = t[i-2]
      y1 = y[i-2,:]
      t2 = t[i-1]
      y2 = y[i-1,:]
      t3 = t[i] 
      y3 = y[i-1,:] + dt * f ( t[i-1], y[i-1,:] )

      y3 = fsolve ( bdf2_residual, y3, args = ( f, dt, t3, y1, y2 ) )

      y[i,:] = y3.copy()

  return t, y

def backward_euler_residual ( yp, f, to, yo, tp ):

#*****************************************************************************80
#
## backward_euler_residual() evaluates the backward Euler residual.
#
#  Discussion:
#
#    We are seeking a value YP defined by the implicit equation:
#
#      YP = YO + ( TP - TO ) * F ( TP, YP )
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    20 October 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real yp: the estimated solution value at the new time.
#
#    function f: evaluates the right hand side of the ODE.  
#
#    real to, yo: the old time and solution value.
#
#    real tp: the new time.
#
#  Output:
#
#    real value: the residual.
#
  value = yp - yo - ( tp - to ) * f ( tp, yp )

  return value

def bdf2_residual ( y3, f, dt, t3, y1, y2 ):

#*****************************************************************************80
#
## bdf2_residual() evaluates the BDF2 residual.
#
#  Discussion:
#
#    We are seeking a value Y3 defined by the implicit equation:
#
#      3 Y3 - 4 Y2 + Y1 = 2 * DT * F ( T3, Y3 )
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 October 2021
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real y3: the solution at the new (estimated) times.
#
#    function handle f: evaluates the right hand side of the ODE.  
#
#    real dt: the time step size
#
#    real t3: the new time
#
#    real y1, y2, y3: the solution at old, current times.
#
#  Output:
#
#    real value: the residual.
#
  value = 3.0 * y3 - 4.0 * y2 + y1 - 2.0 * dt * f ( t3, y3 )

  return value

def bdf2_test ( ):

#*****************************************************************************80
#
## bdf2_test() tests bdf2().
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 May 2022
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform
  import scipy as sp

  print ( '' )
  print ( 'bdf2_test():' )
  print ( '  python version: ' + platform.python_version ( ) )
  print ( '  numpy version:  ' + np.version.version )
  print ( '  scipy version:  ' + sp.version.version )
  print ( '  Test bdf2().' )

  tspan = np.array ( [ 0.0, 5.0 ] )
  y0 = np.array ( [ 5000, 100 ] )
  n = 200
  predator_prey_bdf2 ( tspan, y0, n )

  tspan = np.array ( [ 0.0, 1.0 ] )
  y0 = np.array ( [ 0.0 ] )
  n = 27
  stiff_bdf2 ( tspan, y0, n )
#
#  Terminate.
#
  print ( '' )
  print ( 'bdf2_test():' )
  print ( '  Normal end of execution.' )
  print ( '' )

  return

def predator_prey_bdf2 ( tspan, y0, n ):

#*****************************************************************************80
#
## predator_prey_bdf2(): solve using bdf2().
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 May 2022
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real tspan[2]: the time span
#
#    real y0[2]: the initial condition.
#
#    integer n: the number of steps to take.
#
  import matplotlib.pyplot as plt
  import numpy as np

  print ( '' )
  print ( 'predator_prey_bdf2():' )
  print ( '  A pair of ordinary differential equations for a population' )
  print ( '  of predators and prey are solved using the BDF2,' )
  print ( '  solving the implicit function using fsolve().' )
  print ( '' )
  print ( '  The exact solution shows periodic behavior, with a fixed' )
  print ( '  period and amplitude.' )

  f = predator_prey_deriv
  t, y = bdf2 ( f, tspan, y0, n )
#
#  Plot the solution.
#
  plt.clf ( )
  plt.plot ( y[:,0], y[:,1], 'r-', linewidth = 2 )
  plt.grid ( True )
  plt.xlabel ( '<--- Prey --->' )
  plt.ylabel ( '<--- Predators --->' )
  plt.title ( 'predator prey ODE solved by bdf2()' )
  filename = 'predator_prey_bdf2.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "' + filename + '"' )
  plt.show ( block = False )
  plt.close ( )

  return

def predator_prey_deriv ( t, y ):

#*****************************************************************************80
#
## predator_prey_deriv() evaluates the right hand side of the system.
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    George Lindfield, John Penny,
#    Numerical Methods Using MATLAB,
#    Second Edition,
#    Prentice Hall, 1999,
#    ISBN: 0-13-012641-1,
#    LC: QA297.P45.
#
#  Input:
#
#    real T, the current time.
#
#    real Y[2], the current solution variables, rabbits and foxes.
#
#  Output:
#
#    real DYDT[2], the right hand side of the 2 ODE's.
#
  import numpy as np

  r = y[0]
  f = y[1]

  drdt =    2.0 * r - 0.001 * r * f
  dfdt = - 10.0 * f + 0.002 * r * f

  dydt = np.array ( [ drdt, dfdt ] )

  return dydt

def stiff_bdf2 ( tspan, y0, n ):

#*****************************************************************************80
#
## stiff_bdf2() uses BDF2 on the stiff ODE.
#
#  Discussion:
#
#    fsolve() is used to solve the backward Euler equation.
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 May 2022
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real TSPAN(2): the first and last times.
#
#    real Y0: the initial condition.
#
#    integer N: the number of steps to take.
#
  import matplotlib.pyplot as plt
  import numpy as np

  print ( '' )
  print ( 'stiff_bdf2()' )
  print ( '  Use bdf2() to solve the stiff ODE.' )

  t1, y1 = bdf2 ( stiff_deriv, tspan, y0, n )

  t2 = np.linspace ( tspan[0], tspan[1], 101 )
  y2 = stiff_exact ( t2 )

  plt.plot ( t1, y1, 'ro-', linewidth = 3 )
  plt.plot ( t2, y2, 'b-', linewidth = 3 )
  plt.grid ( True )
  plt.xlabel ( '<-- t -->' )
  plt.ylabel ( '<-- y(t) -->' )
  plt.title ( 'stiff bdf2: time plot' )
  plt.legend ( [ 'Computed', 'Exact' ] )
  filename = 'stiff_bdf2.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "' + filename + '"' )
  plt.show ( block = False )
  plt.close ( )

  return

def stiff_deriv ( t, y ):

#*****************************************************************************80
#
## stiff_deriv() evaluates the right hand side of the stiff equation.
#
#  Discussion:
#
#    y' = 50 * ( cos(t) - y )
#    y(0) = 0
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    06 February 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real T, Y: the time and solution value.
#
#  Output:
#
#    real DYDT: the derivative value.
#
  import numpy as np

  dydt = 50.0 * ( np.cos ( t ) - y )

  return dydt

def stiff_exact ( t ):

#*****************************************************************************80
#
## stiff_exact() evaluates the exact solution of the stiff equation.
#
#  Discussion:
#
#    y' = 50 * ( cos(t) - y )
#    y(0) = 0
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    06 February 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real T(:): the evaluation times.
#
#  Output:
#
#    real Y(:): the exact solution values.
#
  import numpy as np

  value = 50.0 * ( np.sin ( t ) + 50.0 * np.cos(t) \
    - 50.0 * np.exp ( - 50.0 * t ) ) / 2501.0

  return value

def timestamp ( ):

#*****************************************************************************80
#
## timestamp() prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the MIT license. 
#
#  Modified:
#
#    21 August 2019
#
#  Author:
#
#    John Burkardt
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return

if ( __name__ == '__main__' ):
  timestamp ( )
  bdf2_test ( )
  timestamp ( )

y


class DescriptorDAEDataset(Dataset):
    def __init__(
        self,
        name: str,
        *,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        E: Any | None = None,
        A: Any | None = None,
        B: Any | None = None,
        y0: list[float] | np.ndarray | None = None,
        u: list[float] | np.ndarray | None = None,
        source_name: str | None = None,
        t_max: float = 1.0,
        step: float = 0.01,
        ref_meta: dict[str, Any] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []
        self.E = E
        self.A = A
        self.B = B
        self.y0 = None if y0 is None else _float_array(y0)
        self.u = None if u is None else _float_array(u)
        self.source_name = source_name
        self.t_max = t_max
        self.step = step
        self.ref_meta = ref_meta or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self._name

    @property
    def description(self) -> str:
        if self._description is not None:
            return self._description
        if self.source_name is not None:
            return f"SLICOT descriptor-system DAE dataset {self.source_name}."
        return "Descriptor-system DAE dataset."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705.10003707</concept_id>
<concept_desc>Mathematics of computing~Solvers</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003716</concept_id>
<concept_desc>Mathematics of computing~Differential equations</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata["step"] = self.step
        metadata["t_max"] = self.t_max
        if self.source_name is not None:
            metadata["source_name"] = self.source_name
        return metadata


def _descriptor_instance(
    dataset: DescriptorDAEDataset,
    *,
    A: Any,
    B: Any,
    E: Any | None = None,
    source_meta: dict[str, Any] | None = None,
) -> DataInstance:
    A = _square_matrix("A", A)
    if E is None and scipy_sparse.issparse(A):
        E = scipy_sparse.eye(A.shape[0], dtype=np.float64, format="coo")
    elif E is None:
        E = np.eye(A.shape[0], dtype=np.float64)
    else:
        E = _square_matrix("E", E)

    B = _input_matrix(B, A.shape[0])
    if E.shape != A.shape:
        raise ValueError(f"E shape {E.shape} must match A shape {A.shape}.")

    input_count = B.shape[1]
    y0 = np.zeros(A.shape[0], dtype=np.float64) if dataset.y0 is None else dataset.y0
    u = np.ones(input_count, dtype=np.float64) if dataset.u is None else dataset.u
    if y0.shape != (A.shape[0],):
        raise ValueError(f"y0 must have shape {(A.shape[0],)}, got {y0.shape}.")
    if u.shape != (input_count,):
        raise ValueError(f"u must have shape {(input_count,)}, got {u.shape}.")

    meta = {
        "span": (0.0, dataset.t_max),
        "step": dataset.step,
        "y0": y0.tolist(),
        "input": u.tolist(),
        "order": A.shape[0],
        "input_count": input_count,
    }
    if dataset.source_name is not None:
        meta["source_name"] = dataset.source_name
    if source_meta is not None:
        meta["source"] = {
            key: value
            for key, value in source_meta.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }

    return DataInstance(
        inputs=[
            _binsparse_matrix(E),
            _binsparse_matrix(A),
            _binsparse_matrix(B),
            _binsparse_matrix(-A),
            _binsparse_matrix(E),
        ],
        meta=meta,
        ref_meta=dataset.ref_meta,
    )


class _DescriptorDAEGenerator(Generator[DescriptorDAEDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705.10003707</concept_id>
<concept_desc>Mathematics of computing~Solvers</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003716</concept_id>
<concept_desc>Mathematics of computing~Differential equations</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Willow Ahrens", "willow.marie.ahrens@gmail.com")]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to help organize this benchmark module and checks."


class DescriptorDAETestGenerator(_DescriptorDAEGenerator):
    @property
    def name(self) -> str:
        return "dae_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "DAE Test Data Generator"

    @property
    def description(self) -> str:
        return "Inlined descriptor systems for DAE solver correctness tests."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def motivation(self) -> str:
        return "Uses a small singular-mass descriptor system to verify DAE steps."

    @property
    def datasets(self) -> list[DescriptorDAEDataset]:
        return [
            DescriptorDAEDataset(
                "test_slicot_descriptor_2",
                pretty_name="Tiny Descriptor DAE",
                description="Two-variable index-1 descriptor system.",
                suites=["test", "trace"],
                E=np.array([[1.0, 0.0], [0.0, 0.0]]),
                A=np.array([[-2.0, 1.0], [1.0, -1.0]]),
                B=np.array([[1.0], [0.0]]),
                y0=[0.0, 0.0],
                u=[1.0],
                t_max=0.4,
                step=0.1,
                ref_meta={
                    "check_discrete_residual": True,
                    "check_jacobians": True,
                    "residual_tol": 1e-7,
                },
            )
        ]

    def generate(self, dataset: DescriptorDAEDataset) -> DataInstance:
        if dataset.A is None or dataset.B is None:
            raise ValueError("DAE test datasets must define A and B.")
        return _descriptor_instance(dataset, A=dataset.A, B=dataset.B, E=dataset.E)


class SlicotDAEGenerator(_DescriptorDAEGenerator):
    @property
    def name(self) -> str:
        return "slicot_dae_inputs"

    @property
    def pretty_name(self) -> str:
        return "SLICOT DAE Data Generator"

    @property
    def description(self) -> str:
        return "Loads SLICOT descriptor-system benchmarks for implicit DAE solvers."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="SLICOT benchmark examples for model reduction",
                authors=[],
                url=slicot.SLICOT_BENCHMARK_PAGE_URL,
            )
        ]

    @property
    def motivation(self) -> str:
        return (
            "SLICOT's MNA examples are descriptor systems from circuit simulation, "
            "where implicit DAE steps solve linear systems built from explicit "
            "residual Jacobians."
        )

    @property
    def datasets(self) -> list[DescriptorDAEDataset]:
        return [
            DescriptorDAEDataset(
                problem.name,
                source_name=problem.mat_filename,
                pretty_name=f"SLICOT {problem.title}",
                description=f"SLICOT {problem.description or problem.title}.",
                suites=["standard"],
                t_max=0.01,
                step=0.01,
            )
            for problem in slicot.SLICOT_PROBLEMS
            if problem.mat_filename in {"MNA_1.mat", "MNA_4.mat"}
        ]

    def generate(self, dataset: DescriptorDAEDataset) -> DataInstance:
        if dataset.source_name is None:
            raise ValueError("SLICOT DAE datasets must define source_name.")
        variables, source_meta = slicot.load_slicot_problem(dataset.source_name)
        A = _mat_variable(variables, "A")
        B = _mat_variable(variables, "B")
        E = _mat_variable(variables, "E", required=False)
        return _descriptor_instance(dataset, A=A, B=B, E=E, source_meta=source_meta)


def _mat_variable(
    variables: dict[str, Any],
    name: str,
    *,
    required: bool = True,
) -> Any | None:
    for key, value in variables.items():
        if key.lower() == name.lower():
            return value
    if required:
        raise ValueError(f"SLICOT MAT file is missing {name!r}.")
    return None


class _DescriptorDAEBenchmark(Benchmark):
    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705.10003707</concept_id>
<concept_desc>Mathematics of computing~Solvers</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003716</concept_id>
<concept_desc>Mathematics of computing~Differential equations</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Willow Ahrens", "willow.marie.ahrens@gmail.com")]

    @property
    def references(self) -> list[Ref]:
        return SlicotDAEGenerator().references

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to help organize this benchmark module and checks."

    @property
    def motivation(self) -> str:
        return "Solves descriptor-form DAEs using explicit residual Jacobians."

    @property
    def generators(self) -> list[Generator[DescriptorDAEDataset]]:
        return [DescriptorDAETestGenerator(), SlicotDAEGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if not self._ref_meta:
            return

        E, A, B, jac_y, jac_yp = [_dense_binsparse(item) for item in self._input]
        if self._ref_meta.get("check_jacobians"):
            np.testing.assert_allclose(jac_y, -A)
            np.testing.assert_allclose(jac_yp, E)
        if not self._ref_meta.get("check_discrete_residual"):
            return

        time = to_numpy(self._output[0])
        y = to_numpy(self._output[1])
        yp = to_numpy(self._output[2])
        forcing = B @ _float_array(self._meta["input"])
        tol = self._ref_meta.get("residual_tol", 1e-8)
        assert y.shape == (len(time), A.shape[0])
        assert yp.shape == y.shape
        for step_index in range(1, len(time)):
            residual = jac_yp @ yp[step_index] + jac_y @ y[step_index] - forcing
            assert np.linalg.norm(residual) < tol, (
                f"{self.name} residual too high at step {step_index}: {residual}"
            )


class SlicotDAEBDF(_DescriptorDAEBenchmark):
    @property
    def name(self) -> str:
        return "slicot_dae_bdf"

    @property
    def pretty_name(self) -> str:
        return "SLICOT DAE BDF"

    @property
    def description(self) -> str:
        return "Fixed-step BDF2 solver for linear SLICOT descriptor DAEs."

    def benchmark(self, xp, data, meta):
        return _solve_descriptor_bdf2(data, meta)
