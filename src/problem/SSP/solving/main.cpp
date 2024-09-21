#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <algorithm>
#include <filesystem>
#include <cassert>
#include <chrono>
#include "gurobi_c++.h"
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <chrono>

using namespace Gecode;

// This file solves RCPSP instances using Constraint Programming in a branch-and-bound fashion, where bounds are computed using Lagrangian Decomposition and optimized with Adam with a subgradient procedure (Hà, M.H., Quimper, CG., Rousseau, LM. (2015). General Bounding Mechanism for Constraint Programs).
// Lagrangian Decomposition Subproblems consist in keeping all precedence constraints of the original problem and using only one resource constraint.
// The RCPSP instance file is specified by the user, and the solution is written in an output file by the program.
// Subproblems are solved using Gurobi and the CP solver used is Gecode

GRBEnv env = GRBEnv(true);

// Structure of the subproblems of the RCPSP problem
struct SubProblem {
  int n_res;
  int rci;
  int n_tasks;
  std::vector < int > rri;
  std::vector < std::vector < int >> succ;
  std::vector < int > d;
};

// Instance data
namespace {
  // Instances
  extern
  const int * mknps[];
  // Instance names
  extern
  const char * name[];

  /// A wrapper class for instance data
  class Spec {
    protected:
      /// Raw instance data
      int horizon;
    int n_res; // number of resources
    std::vector < int > rc; // resource capacities

    int n_tasks; // number of tasks
    std::vector < int > d; // task durations
    std::vector < std::vector < int >> rr; // resource requirements
    std::vector < std::vector < int >> succ;
    public:
      /// Return the number of resource constraints
      std::vector < int > resource_requirements(int i) const {
        return rr[i];
      }
    std::vector < std::vector < int >> resource_requirements() const {
      return rr;
    }
    /// Return the number of items
    int nb_resources() const {
      return n_res;
    }

    int nb_tasks() const {
      return n_tasks;
    }

    int get_horizon() const {
      return horizon;
    }
    /// Return the profit of item i (index of item starting at 0)
    int task_duration(int i) const {
      return d[i];
    }
    std::vector < int > task_durations() const {
      return d;
    }
    /// Return the capacity of the resources
    int capacity(int i) const {
      return rc[i];
    }

    std::vector < int > capacity() const {
      return rc;
    }
    std::vector < int > successors(int i) const {
      return (succ[i]);
    }

    std::vector < std::vector < int >> successors() const {
      return (succ);
    }

    protected:
      // read PSPLIB instance from the file
      void readData(const char * filename) {
        std::cout << "readData entered" << std::endl;
        char * filename2 = new char[strlen(filename)];
        for (int i = 0; i < strlen(filename); i++) {

          filename2[i] = filename[i];
        }
        std::cout << "first loop done" << std::endl;
        std::ifstream file(filename2);

        if (!file) {
          std::cerr << "Failed to open file: " << filename2 << std::endl;
          return;
        }

        std::string line;
        for (int i = 0; i < 5; ++i) {
          if (!std::getline(file, line)) {
            std::cerr << "Failed to read line " << i + 1 << std::endl;
            return;
          }
        }
        std::getline(file, line);
        sscanf(line.c_str(), "jobs (incl. supersource/sink ):  %d", & n_tasks);
        if (!std::getline(file, line) || sscanf(line.c_str(), "horizon                       :  %d", & horizon) != 1) {
          std::cout << line << std::endl;
          std::cerr << "Failed to read horizon" << std::endl;
          return;
        }

        if (!std::getline(file, line) || !std::getline(file, line) || sscanf(line.c_str(), "  - renewable                 :  %d", & n_res) != 1) {
          std::cerr << "Failed to read renewable resources" << std::endl;
          return;
        }

        int t;
        if (!std::getline(file, line) || sscanf(line.c_str(), "  - nonrenewable              :  %d", & t) != 1 || t != 0) {
          std::cerr << "Failed to read nonrenewable resources" << std::endl;
          return;
        }

        if (!std::getline(file, line) || sscanf(line.c_str(), "  - doubly constrained        :  %d", & t) != 1 || t != 0) {
          std::cerr << "Failed to read doubly constrained resources" << std::endl;
          return;
        }

        n_tasks -= 2;
        rc.resize(n_res);
        d.resize(n_tasks + 1);
        rr.resize(n_res);

        for (int i = 0; i < 8; ++i) {
          if (!std::getline(file, line)) {
            std::cerr << "Failed to read line " << i + 1 << std::endl;
            return;
          }
        }

        for (int i = 0; i < n_tasks + 1; ++i) {
          if (!std::getline(file, line)) {
            std::cerr << "Failed to read task data" << std::endl;
            return;
          }

          std::istringstream iss(line);
          int id, mode, nbSucc;
          iss >> id >> mode >> nbSucc;

          std::vector < int > successors;
          for (int j = 0; j < nbSucc; ++j) {
            int succId;
            iss >> succId;
            succId -= 2;
            successors.push_back(succId);
          }
          succ.push_back(successors);
        }
        for (int i = 0; i < 4; ++i) {
          if (!std::getline(file, line)) {
            std::cerr << "Failed to read line " << i + 1 << std::endl;
            return;
          }
        }

        for (int i = -1; i <= n_tasks + 1; ++i) {
          if (!std::getline(file, line)) {
            std::cerr << "Failed to read task duration and resource requirements" << std::endl;
            return;
          }

          std::istringstream iss(line);
          int id, mode, dur;
          iss >> id >> mode >> dur;

          if (i == -1 || i == n_tasks + 1) {
            assert(dur == 0);
          } else {
            d[i] = dur;
            for (int j = 0; j < n_res; ++j) {
              int res;
              iss >> res;
              rr[j].push_back(res);
            }
          }
        }
        n_tasks += 1;
        for (int i = 0; i < 3; ++i) {
          if (!std::getline(file, line)) {
            std::cerr << "Failed to read line " << i + 1 << std::endl;
            return;
          }
        }

        std::istringstream iss(line);
        for (int j = 0; j < n_res; ++j) {
          iss >> rc[j];
        }

        file.close();
      } /// Initialize
    public: Spec(const char * filename) {
      readData(filename);
    }
  };
}

// Entered by the user
class OptionsRCPSP: public InstanceOptions {
  public: bool activate_bound_computation;
  int K;
  float learning_rate;
  float init_value_multipliers;
  std::ofstream * outputFile;
  const char * filename;
  OptionsRCPSP(const char * s, bool activate_bound_computation0, int K0, float learning_rate0, float init_value_multipliers0, std::ofstream * outputFile_0,
    const char * filename0): InstanceOptions(s),
  activate_bound_computation(activate_bound_computation0),
  K(K0),
  learning_rate(learning_rate0),
  init_value_multipliers(init_value_multipliers0),
  outputFile(outputFile_0),
  filename(filename0) {}
};

//Main class where the solving is carried out
class RCPSP: public IntMinimizeSpace {
  protected: const Spec spec; // Specification of the instance
  IntVarArray s; // start times
  IntVarArray e; // end times
  bool activate_bound_computation;
  int K;
  float learning_rate;
  float init_value_multipliers;
  std::vector < int > order_branching; // Order of the items to branch on
  std::vector < std::vector < float >> multipliers; // Lagrangian multipliers shared between the nodes
  std::ofstream * outputFileMK;
  public: class NoneMin: public Brancher {
    protected: ViewArray < Int::IntView > s;
    // choice definition
    class PosVal: public Choice {
      public: int pos;int val;
      PosVal(const NoneMin & b, int p, int v): Choice(b, 2),
      pos(p),
      val(v) {}
      virtual void archive(Archive & e) const {
        Choice::archive(e);
        e << pos << val;
      }
    };
    public: NoneMin(Home home, ViewArray < Int::IntView > & s0): Brancher(home),
    s(s0) {}
    static void post(Home home, ViewArray < Int::IntView > & s0) {
      (void) new(home) NoneMin(home, s0);
    }
    virtual size_t dispose(Space & home) {
      (void) Brancher::dispose(home);
      return sizeof( * this);
    }
    NoneMin(Space & home, NoneMin & b): Brancher(home, b) {
      s.update(home, b.s);
    }
    virtual Brancher * copy(Space & home) {
      return new(home) NoneMin(home, * this);
    }
    // status
    virtual bool status(const Space & home) const {
      for (int i = 0; i < s.size(); i++)
        if (!s[i].assigned())
          return true;
      return false;
    }
    // choice
    virtual Choice * choice(Space & home) {
      for (int i = 0; true; i++) {
        int index = static_cast < RCPSP & > (home).order_branching[i];
        if (!s[index].assigned()) {
          return new PosVal( * this, index, s[index].min());
        }
      }
      GECODE_NEVER;
      return NULL;
    }
    virtual Choice * choice(const Space & , Archive & e) {
      int pos, val;
      e >> pos >> val;
      return new PosVal( * this, pos, val);
    }
    // commit
    virtual ExecStatus commit(Space & home,
      const Choice & c,
        unsigned int a) {
      const PosVal & pv = static_cast <
        const PosVal & > (c);
      int pos = pv.pos, val = pv.val;
      if (a == 0) {
        ExecStatus temp = me_failed(s[pos].eq(home, val)) ? ES_FAILED : ES_OK;
        //std::cout << "a :" << a << std::endl;
        static_cast < RCPSP & > (home).more();
        return temp;
      } else {
        ExecStatus temp = me_failed(s[pos].nq(home, val)) ? ES_FAILED : ES_OK;
        //std::cout << "a :" << a << std::endl;
        static_cast < RCPSP & > (home).more();
        return temp;
      }
    }
    // print
    virtual void print(const Space & home,
      const Choice & c,
        unsigned int a,
        std::ostream & o) const {
      const PosVal & pv = static_cast <
        const PosVal & > (c);
      int pos = pv.pos, val = pv.val;
      if (a == 0)
        o << "x[" << pos << "] = " << val;
      else
        o << "x[" << pos << "] != " << val;
    }
  };

  void nonemin(Home home,
    const IntVarArgs & s) {
    if (home.failed()) return;
    ViewArray < Int::IntView > y(home, s);
    NoneMin::post(home, y);
  }

  /// Actual model
  RCPSP(const OptionsRCPSP & opt): IntMinimizeSpace(),
  spec(opt.filename),
  s( * this, spec.nb_tasks(), 0, spec.get_horizon()),
  e( * this, spec.nb_tasks(), 0, spec.get_horizon()),
  outputFileMK(opt.outputFile) {
    int n = spec.nb_tasks(); // The number of items
    int m = spec.nb_resources(); // The number of constraints
    int durations[n]; // The profit of the items
    int capacities[m]; // The capacities of the resources
    this -> activate_bound_computation = opt.activate_bound_computation; // Activate the bound computation at each node
    this -> K = opt.K; // The number of iteration to find the optimal multipliers
    this -> learning_rate = opt.learning_rate; // The learning rate to update the multipliers
    this -> init_value_multipliers = opt.init_value_multipliers; // The starting value of the multipliers

    std::vector < std::vector < float >> v; // help to initialize the local handle which will contain the multipliers and be shared between the nodes
    v.resize(n);
    order_branching.resize(n);
    for (int i = 0; i < n; ++i) {
      v[i].resize(m);
      float sum = 0;
      for (int j = 1; j < m; ++j) {
        float rand_num = std::rand();
        rand_num = rand_num / RAND_MAX;
        v[i][j] = init_value_multipliers * 2 * (rand_num - 0.5);
        sum += v[i][j];
      }
      v[i][0] = sum;
    }
    this -> multipliers = v;
    for (int i = 0; i < n; i++) {
      durations[i] = spec.task_duration(i);
    }

    for (int i = 0; i < m; i++) {
      capacities[i] = spec.capacity(i);
    }
    IntVarArgs s2;
    IntVarArgs e2;
    IntVarArgs durations2;
    std::vector < BoolVar > a;
    a.resize(n);
    for (int i = 0; i < n; i++) {
      durations2 << expr( * this, spec.task_duration(i));
      s2 << expr( * this, s[i]);
      e2 << expr( * this, e[i]);
      BoolVar b( * this, 1, 1);
      a[i] = b;
    }
    const BoolVarArgs & mandatory(a);
    for (int i = 0; i < m; i++) {
      capacities[i] = spec.capacity(i);
    }
    for (int i = 0; i < n; i++) {
      order_branching[i] = i;
    }
    const IntVarArgs & s3 = s2;
    const IntVarArgs & e3 = e2;
    const IntVarArgs & durations3 = durations2;
    //for (int i=0;i<m;i++) {
    for (int i = 0; i < m; i++) {
      IntVar c( * this, spec.capacity(i), spec.capacity(i));
      IntArgs u3(spec.resource_requirements(i));
      cumulative( * this, c, s3, durations3, e3, u3, mandatory);

    }
    //precedence constraints
    for (int i = 0; i < n; i++) {
      std::vector < int > vec = spec.successors(i);
      for (int k = 0; k < vec.size(); k++) {
        IntVarArgs diff;
        diff << expr( * this, e[i] - s[vec[k]]);
        linear( * this, diff, IRT_LQ, 0);
      }
    }
    //linking start times and ending times
    for (int i = 0; i < n; i++) {
      IntVarArgs diff;
      diff << expr( * this, e[i] - s[i] - spec.task_duration(i));
      linear( * this, diff, IRT_EQ, 0);
    }
    if (activate_bound_computation) {
      nonemin( * this, s);
    } else {
      branch( * this, s, INT_VAR_NONE(), INT_VAL_MIN());
    }
  }

  void compare(const Space & s, std::ostream & os) const {}

  void more(void) { // compute the bound at each node after every branching, the optimization is carried out with Adam
    float copy_learning_rate = learning_rate;
    int nb_tasks = spec.nb_tasks();
    int nb_resources = spec.nb_resources();
    std::vector < std::vector < int >> succ = spec.successors();
    std::vector < int > capacities = spec.capacity();
    std::vector < int > durations = spec.task_durations();
    int rows = nb_tasks;
    int cols = nb_resources;
    //default adam values
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;

    // store the value of the variable in the solution during the dynamic programming algo to update the multipliers
    int ** value_var_solution = new int * [rows];
    float ** m = new float * [rows];
    float ** v = new float * [rows];

    // init value_var_solution with 0
    for (int i = 0; i < rows; ++i) {
      value_var_solution[i] = new int[cols];
      m[i] = new float[cols];
      v[i] = new float[cols];
      for (int j = 0; j < cols; ++j) {
        value_var_solution[i][j] = 0;
        m[i][j] = 0;
        v[i][j] = 0;
      }
    }

    float final_bound = 0;
    float bound_test[K];

    std::vector < int > not_fixed_variables;
    std::vector < int > fixed_variables;
    bool * is_fixed = new bool[nb_tasks];

    for (int k = 0; k < nb_tasks; k++) {
      if (s[k].size() >= 2) {
        not_fixed_variables.push_back(k);
        is_fixed[k] = false;
      } else {
        is_fixed[k] = true;
        fixed_variables.push_back(k);
      }
    }
    int size_unfixed = not_fixed_variables.size();
    for (int k = 1; k <= K; k++) {
      float bound_iter = 0.0 f;
      for (int id_subproblem = 0; id_subproblem < nb_resources; id_subproblem++) { // iterate on all the constraints (=subproblems of the rcpsp problem)
        std::vector < int > rr = spec.resource_requirements(id_subproblem);
        //paramètres du subproblem
        float bound = mip2(spec.get_horizon(),
          capacities[id_subproblem],
          //spec.capacity(),
          s,
          rr,
          //spec.resource_requirements(),
          succ,
          durations,
          nb_tasks,
          nb_resources,
          id_subproblem,
          value_var_solution,
          is_fixed,
          multipliers);
        bound_iter += bound; // sum all the bound of the rcpsp sub-problem to update the multipliers
      }
      final_bound = std::max(final_bound, bound_iter);
      bound_test[k] = bound_iter;
      for (int i = 0; i < rows; ++i) {
        float sum = 0;
        for (int j = 1; j < cols; ++j) {

          float gradient = -(value_var_solution[i][0] - value_var_solution[i][j]);
          m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradient;

          // Update biased second moment estimate
          v[i][j] = beta2 * v[i][j] + gradient * gradient * (1 - beta2);

          // Compute bias-corrected first moment estimate
          float m_hat = m[i][j] / (1 - std::pow(beta1, k));

          // Compute bias-corrected second moment estimate
          float v_hat = v[i][j] / (1 - std::pow(beta2, k));

          multipliers[i][j] = multipliers[i][j] - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
          //std::cout<<learning_rate * m_hat / (std::sqrt(v_hat) + epsilon)<<" "<<m[i][j]<<" "<<m_hat<<" "<<v[i][j]<<"/";
          sum += multipliers[i][j];
        }
        multipliers[i][0] = sum;

      }
    }

    // We impose the constraint z >= final_bound
    rel( * this, s[nb_tasks - 1] >= final_bound);

    learning_rate = copy_learning_rate;

    for (int i = 0; i < rows; ++i) {
      delete[] value_var_solution[i];
      delete[] m[i];
      delete[] v[i];
    }
    delete[] is_fixed;
    delete[] m;
    delete[] v;
    delete[] value_var_solution;
  }

  static void post(Space & home) {
    static_cast < RCPSP & > (home).more();
  }

  /// Return cost
  virtual IntVar cost(void) const {
    return s[spec.nb_tasks()];
  }
  /// Constructor for cloning \a s
  RCPSP(RCPSP & rcp): IntMinimizeSpace(rcp),
  spec(rcp.spec) {
    s.update( * this, rcp.s);
    e.update( * this, rcp.e);
    this -> order_branching = rcp.order_branching;
    this -> activate_bound_computation = rcp.activate_bound_computation;
    this -> K = rcp.K;
    this -> learning_rate = rcp.learning_rate;
    this -> init_value_multipliers = rcp.init_value_multipliers;
    this -> multipliers = rcp.multipliers;
    this -> outputFileMK = rcp.outputFileMK;
  }
  /// Copy during cloning
  virtual Space *
  copy(void) {
    return new RCPSP( * this);
  }

  virtual void constrain(const Space & _b) { // compute the bound at each leaf node giving a solution
    const RCPSP & b = static_cast <
      const RCPSP & > (_b);

    // We impose the constraint z >= current sol
    rel( * this, s[spec.nb_tasks() - 1] < b.s[spec.nb_tasks() - 1]);
  }

  //Solve subproblems (that is problem with only one resource constraint) : first MIP formulation
  float mip(int horizon,
    int rci,
    IntVarArray s,
    std::vector < int > rri,
    std::vector < std::vector < int >> succ,
    std::vector < int > d,
    int nb_tasks,
    int nb_resources,
    int idx_constraint,
    int ** value_var_solution,
    bool * is_fixed,
    std::vector < std::vector < float >> multipliers) {
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_IntParam_Method, 1);
    model.set(GRB_IntParam_OutputFlag, 0);
    GRBVar * L = new GRBVar[nb_tasks + 2];
    int maxs = 0;
    for (int i = 0; i < nb_tasks + 2; i++) {
      if (i < nb_tasks) {
        L[i] = model.addVar(s[i].min(), s[i].max(), s[i].min(), GRB_INTEGER);
        maxs = std::max(s[i].max(), maxs);
      } else if (i == nb_tasks) {
        L[i] = model.addVar(0, maxs, 0, GRB_INTEGER);
      } else {
        L[i] = model.addVar(0, maxs, maxs, GRB_INTEGER);
      }
      if ((i < nb_tasks) and(s[i].size() == 1)) {
        std::cout << "Variable " << i << " est fixée à " << s[i].val() << std::endl;
        model.addConstr(L[i] == s[i].val());
      }
    }
    for (int i = 0; i < nb_tasks; i++) {
      model.addConstr(L[nb_tasks] <= L[i]);
      model.addConstr(L[i] + d[i] <= L[nb_tasks + 1]);
    }
    //relations de précédence
    for (int i = 0; i < nb_tasks; i++) {
      for (int j = 0; j < succ[i].size(); j++) {
        model.addConstr(L[i] + d[i] <= L[succ[i][j]]);
      }
    }

    GRBVar * O = new GRBVar[horizon * nb_tasks];
    GRBVar ** OIJ = new GRBVar * [nb_tasks * horizon];
    for (int i = 0; i < nb_tasks; i++) {
      for (int j = 0; j < horizon; j++) {
        O[i * horizon + j] = model.addVar(0, 1, 1, GRB_BINARY);
        OIJ[i * horizon + j] = model.addVars(2, GRB_BINARY);
        model.addGenConstrIndicator(OIJ[i * horizon + j][0], true, L[i] <= j);
        model.addGenConstrIndicator(OIJ[i * horizon + j][0], false, L[i] >= j + 1);
        model.addGenConstrIndicator(OIJ[i * horizon + j][1], true, L[i] - 1 >= j - d[i]);
        model.addGenConstrIndicator(OIJ[i * horizon + j][1], false, L[i] <= j - d[i]);
        const GRBVar * oij2 = OIJ[i * horizon + j];
        model.addGenConstrAnd(O[i * horizon + j], oij2, 2);

      }
    }
    GRBVar * cons = model.addVars(horizon, GRB_INTEGER);
    for (int j = 0; j < horizon; j++) {
      GRBLinExpr e1 = GRBLinExpr();
      for (int i = 0; i < nb_tasks; i++) {
        e1 += O[i * horizon + j] * rri[i];
      }
      model.addConstr(e1 <= rci);
      model.addConstr(cons[j] == e1);
    }
    int sum = 0;
    for (int i = 0; i < nb_tasks; i++) {
      sum += rri[i] * d[i];
    }
    GRBLinExpr obj = GRBLinExpr();
    if (idx_constraint == 0) {
      obj += L[nb_tasks + 1];
      for (int i = 0; i < nb_tasks; i++) {
        obj += L[i] * multipliers[i][0];
      }
    } else {
      for (int i = 0; i < nb_tasks; i++) {
        obj += -L[i] * multipliers[i][idx_constraint];
      }
    }
    model.setObjective(obj, GRB_MINIMIZE);
    model.optimize();
    float bound;
    bound = obj.getValue();
    for (int j = 0; j < nb_tasks; j++) {
      value_var_solution[j][idx_constraint] = L[j].get(GRB_DoubleAttr_X);
    }
    int sum2 = 0;
    for (int i = 0; i < horizon; i++) {
      sum2 += cons[i].get(GRB_DoubleAttr_X);
    }
    if (idx_constraint == 0) {
      bound = L[nb_tasks + 1].get(GRB_DoubleAttr_X);
      for (int i = 0; i < nb_tasks; i++) {
        bound += multipliers[i][0] * L[i].get(GRB_DoubleAttr_X);
      }
    } else {
      bound = 0;
      for (int i = 0; i < nb_tasks; i++) {
        bound += -multipliers[i][idx_constraint] * L[i].get(GRB_DoubleAttr_X);
      }
    }
    std::cout << "bound : " << bound << std::endl;
    return (bound);
  }

  //Solve subproblems : second MIP formulation
  float mip2(int horizon,
    int rci,
    IntVarArray s,
    std::vector < int > rri,
    std::vector < std::vector < int >> succ,
    std::vector < int > d,
    int nb_tasks,
    int nb_resources,
    int idx_constraint,
    int ** value_var_solution,
    bool * is_fixed,
    std::vector < std::vector < float >> multipliers,
    bool complet = false) {
    if (idx_constraint != 0) {
      return (0);
    }
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_IntParam_Method, 1);
    model.set(GRB_IntParam_OutputFlag, 0);
    GRBVar * y = model.addVars(nb_tasks * horizon, GRB_BINARY);
    GRBLinExpr * start_times = new GRBLinExpr[nb_tasks + 1];
    int maxend = 0;
    for (int i = 0; i < nb_tasks; i++) {
      maxend = std::max(maxend, s[i].max());
      GRBLinExpr e = GRBLinExpr();
      start_times[i] = GRBLinExpr();
      for (int t = 0; t < horizon; t++) {
        e += y[i * horizon + t];
        start_times[i] += y[i * horizon + t] * t;
      }
      model.addConstr(e, GRB_EQUAL, 1);
      if ((s[i].size() == 1) and(i < nb_tasks)) {
        model.addConstr(start_times[i] == s[i].val());
        for (int j = 0; j < horizon; j++) {
          if (j != s[i].val()) {
            model.addConstr(y[i * horizon + j] == 0);
          } else {
            model.addConstr(y[i * horizon + j] == 1);
          }
        }
      } else if (i < nb_tasks) {
        model.addConstr(start_times[i] <= s[i].max());
        model.addConstr(s[i].min() <= start_times[i]);
        for (int j = 0; j < horizon; j++) {
          if ((j < s[i].min()) or(j > s[i].max())) {
            model.addConstr(y[i * horizon + j] == 0);
          }
        }
      } else {

      }
    }
    //precedence relations, from the given precedence graph
    for (int i = 0; i < nb_tasks; i++) {
      for (int j = 0; j < succ[i].size(); j++) {
        GRBLinExpr e = GRBLinExpr();
        e = start_times[succ[i][j]] - start_times[i];
        model.addConstr(e, GRB_GREATER_EQUAL, d[i]);
      }
    }
    //resource requirements
    GRBLinExpr ** EE2 = new GRBLinExpr * [horizon];
    for (int t = 0; t < horizon; t++) {
      GRBLinExpr e = GRBLinExpr();
      EE2[t] = new GRBLinExpr[nb_tasks];
      for (int i = 0; i < nb_tasks; i++) {
        EE2[t][i] = GRBLinExpr();
        for (int t2 = t - d[i] + 1; t2 <= t; t2++) {
          if (t2 >= 0) {
            EE2[t][i] += y[i * horizon + t2];
          }
        }
        e += EE2[t][i] * rri[i];
      }
      model.addConstr(e, GRB_LESS_EQUAL, rci);
    }
    GRBLinExpr obj = GRBLinExpr();
    if (idx_constraint == 0) {
      obj += start_times[nb_tasks - 1];
      for (int i = 0; i < nb_tasks; i++) {
        obj += start_times[i] * multipliers[i][0];
      }
    } else {
      for (int i = 0; i < nb_tasks; i++) {
        obj += -start_times[i] * multipliers[i][idx_constraint];
      }
    }
    model.setObjective(obj, GRB_MINIMIZE);
    auto start = std::chrono::high_resolution_clock::now();
    model.optimize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration < double > duration = end - start;
    try {
      float bound = 0;
      for (int i = 0; i < nb_tasks; i++) {
        bool b = false;
        for (int t = 0; t < horizon; t++) {
          if (y[i * horizon + t].get(GRB_DoubleAttr_X) == 1) {
            if ((i == nb_tasks - 1) and(idx_constraint == 0)) {
              bound += t;
              value_var_solution[i][idx_constraint] = t;
              bound += t * multipliers[i][0];
              b = true;
            } else if (b) {
              std::cout << "ERREUR" << std::endl;
            } else if (i < nb_tasks) {
              value_var_solution[i][idx_constraint] = t;
              if (idx_constraint == 0) {
                bound += t * multipliers[i][0];
              } else {
                bound += -t * multipliers[i][idx_constraint];
              }
              b = true;
            }
          }
        }
      }
      delete[] start_times;
      for (int t = 0; t < horizon; t++) {
        delete[] EE2[t];
      }
      delete[] EE2;
      delete[] y;
      return (bound);
    } catch (GRBException e) {
      std::cout << "Pas de solution au sous-problème" << std::endl;
      delete[] start_times;
      for (int t = 0; t < horizon; t++) {
        delete[] EE2[t];
      }
      delete[] EE2;
      delete[] y;
      return (10000);
    }
  }
  //solve an entire problem with the second MIP formulation
  float mip2full(int horizon,
    std::vector < int > rci,
    IntVarArray s,
    std::vector < std::vector < int >> rri,
    std::vector < std::vector < int >> succ,
    std::vector < int > d,
    int nb_tasks,
    int nb_resources,
    int idx_constraint,
    int ** value_var_solution,
    bool * is_fixed,
    std::vector < std::vector < float >> multipliers,
    bool complet = false) {
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_IntParam_Method, 1);
    model.set(GRB_IntParam_OutputFlag, 0);
    GRBVar * y = model.addVars((nb_tasks + 1) * horizon, GRB_BINARY);
    GRBLinExpr * start_times = new GRBLinExpr[nb_tasks + 1];
    for (int i = 0; i < nb_tasks; i++) {
      GRBLinExpr e = GRBLinExpr();
      start_times[i] = GRBLinExpr();
      for (int t = 0; t < horizon; t++) {
        e += y[i * horizon + t];
        start_times[i] += y[i * horizon + t] * t;
      }
      model.addConstr(e, GRB_EQUAL, 1);
      if ((s[i].size() == 1) and(i < nb_tasks)) {
        //std::cout<<"Variable "<<i<<" est fixée à "<<s[i].val()<<std::endl;
        model.addConstr(start_times[i] == s[i].val());
      }
    }
    //relations de précédence
    for (int i = 0; i < nb_tasks; i++) {
      //issues du graphe de précédence
      for (int j = 0; j < succ[i].size(); j++) {
        GRBLinExpr e = GRBLinExpr();
        e = start_times[succ[i][j]] - start_times[i];
        model.addConstr(e, GRB_GREATER_EQUAL, d[i]);
      }
    }
    //resource requirements
    GRBVar * cons = model.addVars(horizon, GRB_INTEGER);
    for (int k = 0; k < nb_resources; k++) {
      for (int t = 0; t < horizon; t++) {
        GRBLinExpr e = GRBLinExpr();
        GRBLinExpr * E2 = new GRBLinExpr[nb_tasks];
        for (int i = 0; i < nb_tasks; i++) {
          E2[i] = GRBLinExpr();
          for (int t2 = t - d[i] + 1; t2 <= t; t2++) {
            if (t2 >= 0) {
              E2[i] += y[i * horizon + t2];
            }
          }
          e += E2[i] * rri[k][i];
        }
        model.addConstr(e, GRB_LESS_EQUAL, rci[k]);
        if ((k == 0) and(t == 0)) {
          model.addConstr(cons[t] == e);
        }
      }
    }
    GRBLinExpr obj = GRBLinExpr();
    if (idx_constraint == 0) {
      obj += start_times[nb_tasks - 1];
      for (int i = 0; i < nb_tasks; i++) {
        obj += start_times[i] * multipliers[i][0];
      }
    } else {
      for (int i = 0; i < nb_tasks; i++) {
        obj += -start_times[i] * multipliers[i][idx_constraint];
      }
    }
    model.setObjective(obj, GRB_MINIMIZE);
    model.optimize();
    int bound;
    bound = obj.getValue();
    for (int i = 0; i < nb_tasks; i++) {
      bool b = false;
      for (int t = 0; t < horizon; t++) {
        if (y[i * horizon + t].get(GRB_DoubleAttr_X) == 1) {
          if (b) {
            std::cout << "ERREUR" << std::endl;
          } else {
            value_var_solution[i][idx_constraint] = t;
            b = true;
          }
        }
      }
    }
    std::cout << "integer bound : " << obj.getValue() << std::endl;
    return (bound);
  }

  //solve a subproblem with the third formulation
  float mip3(int horizon,
    int rci,
    IntVarArray s,
    std::vector < int > rri,
    std::vector < std::vector < int >> succ,
    std::vector < int > d,
    int nb_tasks,
    int nb_resources,
    int idx_constraint,
    int ** value_var_solution,
    bool * is_fixed,
    std::vector < std::vector < float >> multipliers) {
    std::cout << "" << std::endl;
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_IntParam_Method, 1);
    model.set(GRB_IntParam_OutputFlag, 0);
    GRBVar * t = new GRBVar[nb_tasks + 1];
    for (int i = 0; i < nb_tasks + 1; i++) {
      t[i] = model.addVar(0, horizon, 0, GRB_INTEGER);
    }
    GRBVar * x = model.addVars(nb_tasks * (nb_tasks + 1), GRB_BINARY);
    GRBVar * y = model.addVars(nb_tasks * (nb_tasks + 1), GRB_BINARY);
    GRBLinExpr * r = new GRBLinExpr[nb_tasks + 1];
    for (int e = 0; e < nb_tasks + 1; e++) {
      for (int f = e + 1; f < nb_tasks + 1; f++) {
        for (int i = 0; i < nb_tasks; i++) {
          model.addConstr(t[f] >= t[e] + d[i] * x[i * (nb_tasks + 1) + e] - d[i] * (1 - y[i * (nb_tasks + 1) + f]));
        }
      }
    }
    model.addConstr(t[0] == 0);
    for (int e = 0; e < nb_tasks; e++) {
      model.addConstr(t[e + 1] >= t[e]);
    }
    for (int i = 0; i < nb_tasks; i++) {
      GRBLinExpr exp = GRBLinExpr();
      for (int e = 0; e < nb_tasks + 1; e++) {
        exp += x[i * (nb_tasks + 1) + e];
      }
      model.addConstr(exp == 1);
      GRBLinExpr exp2 = GRBLinExpr();
      for (int e = 0; e < nb_tasks + 1; e++) {
        exp2 += y[i * (nb_tasks + 1) + e];
      }
      model.addConstr(exp2 == 1);
    }
    for (int i = 0; i < nb_tasks; i++) {
      for (int j = 0; j < succ[i].size(); j++) {
        for (int e = 0; e < nb_tasks; e++) {
          GRBLinExpr exp = GRBLinExpr();
          for (int e2 = 0; e2 <= e - 1; e2++) {
            exp += x[succ[i][j] * (nb_tasks + 1) + e2];
          }
          for (int e2 = e; e2 <= nb_tasks; e2++) {
            exp += y[i * (nb_tasks + 1) + e2];
          }
          model.addConstr(exp <= 1);
        }
      }
      for (int e = 0; e <= nb_tasks; e++) {
        GRBLinExpr exp = GRBLinExpr();
        for (int e2 = 0; e2 <= e; e2++) {
          exp += y[i * (nb_tasks + 1) + e2];
        }
        for (int e2 = e; e2 <= nb_tasks; e2++) {
          exp += x[i * (nb_tasks + 1) + e2];
        }
        model.addConstr(exp <= 1);
      }
    }

    r[0] = GRBLinExpr();
    for (int i = 0; i < nb_tasks; i++) {
      r[0] += rri[i] * x[i * (nb_tasks + 1)];
    }
    model.addConstr(r[0] <= rci);
    model.addConstr(r[0] >= 0);
    for (int e = 1; e < nb_tasks + 1; e++) {
      r[e] = GRBLinExpr();
      r[e] += r[e - 1];
      for (int i = 0; i < nb_tasks; i++) {
        r[e] += rri[i] * x[i * (nb_tasks + 1) + e];
        r[e] += -rri[i] * y[i * (nb_tasks + 1) + e];
      }
      model.addConstr(r[e] <= rci);
      model.addConstr(r[e] >= 0);
    }

    GRBLinExpr obj = GRBLinExpr();
    if (idx_constraint == 0) {
      obj += t[nb_tasks];
      for (int i = 0; i < nb_tasks; i++) {
        obj += t[i] * multipliers[i][0];
      }
    } else {
      for (int i = 0; i < nb_tasks; i++) {
        obj += -t[i] * multipliers[i][idx_constraint];
      }
    }
    model.setObjective(obj, GRB_MINIMIZE);
    std::cout << "lets optimize" << std::endl;
    model.optimize();
    int bound;
    bound = obj.getValue();
    std::cout << "start times :" << std::endl;
    for (int i = 0; i < nb_tasks; i++) {
      value_var_solution[i][idx_constraint] = t[i].get(GRB_DoubleAttr_X);
      std::cout << t[i].get(GRB_DoubleAttr_X) << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "x 15 : ";
    for (int t = 0; t < nb_tasks + 1; t++) {
      /*
      if (x[20*(nb_tasks+1)+t].get(GRB_DoubleAttr_X)==1) {
          std::cout<<t<<std::endl;
      } */
      std::cout << x[15 * (nb_tasks + 1) + t].get(GRB_DoubleAttr_X) << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "y 15 : ";
    for (int t = 0; t < nb_tasks + 1; t++) {
      /*
      if (y[20*(nb_tasks+1)+t].get(GRB_DoubleAttr_X)==1) {
          std::cout<<t<<std::endl;
      }*/
      std::cout << y[15 * (nb_tasks + 1) + t].get(GRB_DoubleAttr_X) << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "bound : " << bound << std::endl;
    return (bound);
  }
  virtual void print(std::ostream & os) const {
    os << "s: " << s[0] << std::endl;
    os << "x: " << s[spec.nb_tasks() - 1] << std::endl;
  }
};

int main(int argc, char * argv[]) {

  // This code reads input data from a given file specified by a user, solves the rcpsp instance in a branch-and-bound fashion and writes the solution in an output file "test.txt"
  std::srand(std::time(nullptr));
  bool activate_bound_computation;
  if (std::string(argv[2]) == "F") {
    activate_bound_computation = false;
  } else {
    activate_bound_computation = true;
    env.start();
  }
  int K = 200;
  float learning_rate = 0.001;
  float init_value_multipliers = 0;
  std::cout << argv[1] << std::endl;
  std::ofstream * outputFilea = new std::ofstream("test.txt");
  const char * dir = "/home/bessa75/scratch/scratch2/instances/";
  char * filename = new char[strlen(dir) + strlen(argv[1])];
  for (int i = 0; i < strlen(dir) + strlen(argv[1]); i++) {
    if (i < strlen(dir)) {
      filename[i] = dir[i];
    } else {
      filename[i] = argv[1][i - strlen(dir)];
    }
  }
  const char * filename2 = filename;
  OptionsRCPSP opt("RCPSP", activate_bound_computation, K, learning_rate, init_value_multipliers, outputFilea, filename2);
  opt.solutions(0);
  opt.parse(argc, argv);
  IntMinimizeScript::run < RCPSP, BAB, OptionsRCPSP > (opt);
  outputFilea -> close();
  std::cout << "Lines have been written to test.txt" << std::endl;
  return 0;
}

namespace {}
