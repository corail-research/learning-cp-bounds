#include <gecode/driver.hh>
#include <torch/script.h> 

#include <gecode/int.hh>
#include <gecode/minimodel.hh>

#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <random>
#include <cstdlib>
#include <unistd.h>

using namespace Gecode;

// Keep track of the nodes that have been written in the file
static std::unordered_set<std::string> set_nodes;
static torch::jit::script::Module module_1;
static torch::jit::script::Module module_2;
static int compteur_iterations = 0;

// Allow to store the multipliers in a local handle to share them between the nodes
class LI : public LocalHandle {
    protected:
    class LIO : public LocalObject {
    public:
        std::vector<std::vector<std::vector<float>>> multipliers;
        LIO(Space& home, std::vector<std::vector<std::vector<float>>> d) : LocalObject(home), multipliers(d) {}
        LIO(Space& home, LIO& l)
        : LocalObject(home,l), multipliers(l.multipliers) {}
        virtual LocalObject* copy(Space& home) {
        return new (home) LIO(home,*this);
        }
        virtual size_t dispose(Space&) { return sizeof(*this); }
    };

    public:
    LI(Space& home, std::vector<std::vector<std::vector<float>>> d)
        : LocalHandle(new (home) LIO(home,d)) {}
    LI(const LI& li)
        : LocalHandle(li) {}
    LI& operator =(const LI& li) {
        return static_cast<LI&>(LocalHandle::operator =(li));
    }
    std::vector<std::vector<std::vector<float>>> get(void) const {
        return static_cast<LIO*>(object())->multipliers;
    }
    void set(std::vector<std::vector<std::vector<float>>> d) {
        static_cast<LIO*>(object())->multipliers = d;
    }
    };

//TODO: Change the structure for SSP
// Structure of the subproblems of the ssp problem
struct SubProblem {
    std::vector<std::tuple<int, int, int>> transitions_sub;
    std::vector<std::vector<float>> val_sub;
    std::vector<std::vector<int>> domains_sub;
    std::vector<int> states;
    std::vector<int> final_states;
    int idx_constraint;
    };

// Instance data
namespace {
    // Instances
    extern int* mknps[];
    // Instance names
    extern const char* name[];

    /// A wrapper class for instance data
    // TODO: Change the structure for SSP
    class Spec {
    private:
        int* pData;
    protected:
        /// Lower and upper bound
        int l, u;
    public:
        /// Whether a valid specification has been found
        bool valid(void) const {
        return pData != NULL;
        }
        /// Return the number of multi-knapsack constraints
        int nb_constraints(void) const {
        return pData[0];
        }
        int pdatai(int i) const{
            //std::cout<<"i : "<<i<<std::endl;
            return pData[i];
        }
        /// Return the number of items
        int nb_items(void) const {
        return pData[1];
        }

        int nb_values(void) const {
        return pData[2];
        }

        int nb_states(void) const {
        return pData[3];
        }

        int nb_final_states(void) const {
        return pData[4];
        }

        int initial_state(void) const {
        return pData[5];
        }

        int final_state(int i, int nb_final_states) const {
        return pData[6 + i];
        }
        /// Return the profit of item i for value j (index of item starting at 0)
        int profit(int i, int j, int nb_final_states) const {
        return pData[6 + nb_final_states + i * nb_values() + j];
        }
        /// Return 1 if the value j is in the domain of variable i and 0 otherwise
        int value(int i, int j, int nb_final_states, int nb_items, int nb_values) const {
        return pData[6 + nb_final_states + nb_items * nb_values + j * nb_items + i ];
        }
        /// Return the 1 if the transition from state i to state j with value k is possible and 0 otherwise for constraint l
        int transition(int l, int i, int k, int j, int nb_final_states, int nb_items, int nb_constraints, int nb_states, int nb_values) const {
        return pData[6 + nb_final_states + nb_items * nb_values + nb_items * nb_values + k * nb_states + j * nb_values * nb_states + i + l * nb_values * nb_states * nb_states ];
        }
        // /// Return the optimal value (last value in the instance file)
        // int optimal(int nb_items, int nb_constraints) const {
        // return pData[2+nb_items+nb_constraints+nb_constraints*nb_items+1];
        // }
    protected:
        /// Find instance by name \a s
        static int* find(const char* s) {
        for (int i=0; name[i] != NULL; i++)
            if (!strcmp(s,name[i]))
            return mknps[i];
        return NULL;
        }
        /// Compute lower bound
        int clower(void) const {
        int l = 0;
        return l;
        }
        /// Compute upper bound
        int cupper(void) const {
        int u = 500000;
        return u+1;
        }
        int nb_params(void) const {
            return(6 + nb_final_states() + nb_items() * nb_values() + nb_items() * nb_values() + nb_constraints() * nb_items() * nb_states() * nb_values() );
        }
        int* get_pData() const {
            return pData;
        }

    public:
        /// Initialize
        Spec(const Spec& s){
            l=clower();
            u=cupper();
            
            // Allocate memory for a new array
            int* L = new int [s.nb_params()];
            for (int i=0;i<s.nb_params();i++) {
                L[i]=s.pdatai(i);
            }
            pData=L;
        }
        Spec(std::vector<int> problem,const char* s) :l(0),u(0) {
        if (std::strlen(s)==0) {
            int* L = new int [problem.size()];
            for (int i=0;i<problem.size();i++) {
                L[i]=problem[i];
            }
            this->pData=L;
            }
        else {
            this->pData=find(s);
        }
        if (valid()) {
            l = clower(); u = cupper();
        }
        }
        /// Return lower bound
        int lower(void) const {
        return l;
        }
        /// Return upper bound
        int upper(void) const {
        return u;
        }
    };
    }

// Entered by the user
class OptionsSSP: public InstanceOptions {
  public:
  bool activate_bound_computation;
  bool activate_adam;
  bool activate_learning_prediction;
  bool activate_learning_and_adam;
  bool activate_heuristic;
  int K;
  float learning_rate;
  float init_value_multipliers;
  std::ofstream* outputFile;
  bool write_samples;
  const char* s;
  std::vector<int> problem={};
  OptionsSSP(bool activate_bound_computation0,
        bool activate_adam0,
        bool activate_learning_prediction0,
        bool activate_learning_and_adam0,
        bool activate_heuristic0, 
        int K0, float learning_rate0, 
        float init_value_multipliers0, 
        std::ofstream* outputFile_0, 
        std::vector<int> problem0,
        bool write_samples0=true,const char* s0="")
        : InstanceOptions("SSP"), s(s0), 
        activate_bound_computation(activate_bound_computation0),
        activate_adam(activate_adam0),
        activate_learning_prediction(activate_learning_prediction0),  
        activate_learning_and_adam(activate_learning_and_adam0),
        activate_heuristic(activate_heuristic0), 
        K(K0), learning_rate(learning_rate0), 
        init_value_multipliers(init_value_multipliers0), 
        outputFile(outputFile_0),
        write_samples(write_samples0) {
            if (problem0.size()>1) {
                problem=problem0;
            }
        }
};

class SSP : public IntMaximizeSpace {
  protected:
      const Spec spec; // Specification of the instance
      IntVarArray variables; // Decision variables for each item
      IntVar z; // Variable for the objective function
      bool activate_bound_computation;
      bool activate_adam;
      bool activate_learning_prediction;
      bool activate_learning_and_adam;
      bool activate_heuristic;
      int K;
      float learning_rate;
      float init_value_multipliers;
      std::vector<int> order_branching; // Order of the items to branch on
      std::vector<std::vector<std::vector<float>>> multipliers; // Lagrangian multipliers shared between the nodes
      std::ofstream* outputFileMK;
      bool write_samples;
  public:

  class NoneMax : public Brancher {
    protected:
    ViewArray<Int::IntView> variables;
    // choice definition
    class PosVal : public Choice {
      public:
        int pos; int val;
        PosVal(const NoneMax& b, int p, int v)
        : Choice(b,2), pos(p), val(v) {}
        virtual void archive(Archive& e) const {
        Choice::archive(e);
        e << pos << val;
        }
    };
    
    public:
    NoneMax(Home home, ViewArray<Int::IntView>& variables0)
        : Brancher(home), variables(variables0) {}

    static void post(Home home, ViewArray<Int::IntView>& variables) {
      (void) new (home) NoneMax(home,variables);
    }

    virtual size_t dispose(Space& home) {
      (void) Brancher::dispose(home);
      return sizeof(*this);
    }

    NoneMax(Space& home, NoneMax& b)
        : Brancher(home,b) {
        variables.update(home,b.variables);
    }

    virtual Brancher* copy(Space& home) {
        return new (home) NoneMax(home,*this);
    }

    // status
    virtual bool status(const Space& home) const {
        for (int i=0; i<variables.size(); i++)
        if (!variables[i].assigned())
            return true;
        return false;
    }
    // choice
    //TODO: Change the heurisitc for SSP
    virtual Choice* choice(Space& home) {
        bool activate_heuristic=static_cast<SSP&>(home).activate_heuristic;
        for (int i=0; true; i++){
            int index;
        // if (activate_heuristic) {
        //     index = static_cast<SSP&>(home).order_branching[i]; }
        // else {
        //     index=i;
        // }
        index = i;
        if (!variables[index].assigned()){
        return new PosVal(*this,index,variables[index].max());
        } }
        GECODE_NEVER;
        return NULL;
    }

    virtual Choice* choice(const Space&, Archive& e) {
        int pos, val;
        e >> pos >> val;
        return new PosVal(*this, pos, val);
    }

  // commit
    virtual ExecStatus commit(Space& home,
                                const Choice& c,
                                unsigned int a) {
        bool activate_bound=static_cast<SSP&>(home).activate_bound_computation;
        const PosVal& pv = static_cast<const PosVal&>(c);
        int pos=pv.pos, val=pv.val;
        if (a == 0){
        ExecStatus temp = me_failed(variables[pos].eq(home,val)) ? ES_FAILED : ES_OK;
        if (activate_bound)
            static_cast<SSP&>(home).more();
        return temp;
        }
        else{
        ExecStatus temp = me_failed(variables[pos].nq(home,val)) ? ES_FAILED : ES_OK;
        if (activate_bound)
            static_cast<SSP&>(home).more();
        return temp;
        }
    }

    // print
    virtual void print(const Space& home, const Choice& c,
                        unsigned int a,
                        std::ostream& o) const {
        const PosVal& pv = static_cast<const PosVal&>(c);
        int pos=pv.pos, val=pv.val;
        if (a == 0)
        o << "x[" << pos << "] = " << val;
        else
        o << "x[" << pos << "] != " << val;
    }
    };

    void nonemax(Home home, const IntVarArgs& variables) {
    if (home.failed()) return;
    ViewArray<Int::IntView> y(home,variables);
    NoneMax::post(home,y);
    }

  /// Actual model
  SSP(const OptionsSSP& opt)
    :IntMaximizeSpace(),
    spec(opt.problem,opt.s),
    z(*this, spec.lower(), spec.upper()),
    outputFileMK(opt.outputFile),
    variables(*this, spec.nb_items()),
    write_samples(opt.write_samples){
        // TODO: Change the arguments for the SSP
    int n = spec.nb_items();        // The number of items
    int m = spec.nb_constraints();  // The number of constraints
    int Q = spec.nb_states();        // The number of states
    int F = spec.nb_final_states(); // The number of final states
    int V = spec.nb_values();        // The number of values
    int final_states[F];            // The final states
    int profits[n][V];                 // The profit of the items
    int transitions[m][Q][V][Q]; // The transitions of the items between the states (one vector per state)
    this->activate_bound_computation = opt.activate_bound_computation; // Activate the bound computation at each node
    this->activate_adam = opt.activate_adam; // Activate the Adam optimizer to update the multipliers
    this->activate_learning_prediction = opt.activate_learning_prediction; // Activate the learning prediction
    this->activate_learning_and_adam = opt.activate_learning_and_adam; // Activate the learning prediction and the Adam optimizer
    this->activate_heuristic = opt.activate_heuristic; // Activate the heuristic to branch on the items
    this->K = opt.K;                // The number of iteration to find the optimal multipliers
    this->learning_rate = opt.learning_rate; // The learning rate to update the multipliers
    this->init_value_multipliers = opt.init_value_multipliers; // The starting value of the multipliers

    for (int i = 0; i < n; i ++){
      int len = 0;
      std::vector<int> s;
      for (int j = 0; j < V; j++){
        if (spec.value(i, j, F, n, V) == 1){
          s.push_back(j);
          len++;
        }
      }
      int set[len];
      for (int j = 0; j < len; j++){
        set[j] = s[j];
      }
      IntSet c(set, len);
      variables[i] = IntVar(*this, c);
    }

    // print the value from domain of the variables
    // for (int i = 0; i < n; i++) {
    //     std::cout << "Variable " << i << " : " << variables[i] << std::endl;
    // }

    std::vector<std::vector<std::vector<float>>> v; // help to initialize the local handle which will contain the multipliers and be shared between the nodes
    v.resize(n);
    for (int i = 0; i < n; ++i) {
      v[i].resize(m);
      for (int j = 1; j < m; ++j) {
        v[i][j].resize(V);
        for (int k = 0; k < V; ++k) {
         v[i][j][k] = init_value_multipliers;
        }
      }
    }

    for (int i = 0; i < n; ++i) {
      v[i][0].resize(V);
      for (int k = 0; k < V; ++k) {
        float sum = 0.0f;
        for (int j = 1; j < m; ++j) {
          sum += v[i][j][k];
        }
        v[i][0][k] = sum;
      }
    }

    LI multipliers(*this, v);
    this->multipliers = multipliers.get();

    for (int i = 0; i < F; i++) {
      final_states[i] = spec.final_state(i, F);
    }

    // Get the profits, capacities and weights of the items from the instance
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < V; j++) {
        profits[i][j] = spec.profit(i, j, F);
      }
    }

    IntVarArray C = IntVarArray(*this, n);
    for (int i = 0; i < n; i++) {
      int min_profit = profits[i][0];
      int max_profit = profits[i][0];
    
      for (int j = 1; j < V; j++) {
        if (profits[i][j] < min_profit) {
            min_profit = profits[i][j];
        }
        if (profits[i][j] > max_profit) {
            max_profit = profits[i][j];
        }
      }
      C[i] = IntVar(*this, min_profit, max_profit);
    }

    for (int l = 0; l < m; l++) {
      for (int i = 0; i < Q; i++) {
        for (int k = 0; k < V; k++) {
          for (int j = 0; j < Q; j++) {
            transitions[l][i][k][j] = spec.transition(l, i, k, j, F, n, m, Q, V);
          }
        }
      }
    }
    std::vector<IntArgs> profits_x;
    for (int i = 0; i < n; i++) {
    std::vector<int> c(V);
    for (int j = 0; j < V; j++) {
        c[j] = profits[i][j];
    }
    profits_x.push_back(IntArgs(c));
    }
    // The objective function
    for (int i = 0; i < n; i++) {
        element(*this, profits_x[i], variables[i], C[i]);
    }
    z = expr(*this, sum(C));

    // // print the value from domain of the variables
    // for (int i = 0; i < n; i++) {
    //     std::cout << "Variable " << i << " : " << variables[i] << std::endl;
    // }

    std::vector<DFA> dfas;
    std::vector<std::vector<DFA::Transition>> transitions_vectors(m, std::vector<DFA::Transition>());

    for (int l = 0; l < m; l++) {
      // The constraints for the transitions
      int p = 0;
    
      for (int i = 0; i < Q; i++) {
        for (int k = 0; k < V; k++) {
          for (int j = 0; j < Q; j++) {
            if ((transitions[l][i][k][j] == 1)) {
              transitions_vectors[l].push_back(DFA::Transition(i, k, j));
              p++;
            }
          }
        }
      }

      DFA::Transition transitions_set[p+1];

      for (int i = 0; i < p; i++) {
        transitions_set[i] = transitions_vectors[l][i];
      }

      transitions_set[p] = DFA::Transition(-1, 0, 0);

      int f[F+1]; 

      for (int i = 0; i < F; i++) {
        f[i] = final_states[i];
      }

      f[F] = -1;

      dfas.push_back(DFA(spec.initial_state(), transitions_set, f));

      extensional(*this, variables, dfas[l]);
    }

    // print the value from domain of the variables
    // for (int i = 0; i < n; i++) {
    //     std::cout << "Variable " << i << " : " << variables[i] << std::endl;
    // }

    nonemax(*this, variables);
    
    }

    void compare(const Space& s, std::ostream& os) const {
    }

  void more(void) { // compute the bound at each node after every branching
    const int nb_items = spec.nb_items();
    const int nb_constraints = spec.nb_constraints();
    const int nb_values = spec.nb_values();
    const int nb_states = spec.nb_states();
    const int F = spec.nb_final_states();
    const int initial_state = spec.initial_state();
    int val[nb_items];
    int profits[nb_items][nb_values];
    int transitions[nb_constraints][nb_states][nb_values][nb_states];
    float final_fixed_bounds = 0.0f;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;

    int rows = nb_items;
    int cols = nb_constraints;
    //  std::cout << "more" << std::endl;
    // for (int i = 0; i < nb_items; i++) {
    //     std::cout << "Variable " << i << " : " << variables[i] << std::endl;
    // }

    // store the value of the variable in the solution during the dynamic programming algo to update the multipliers
    // int** value_var_solution = new int*[rows];
    std::vector<std::vector<int>> value_var_solution(cols, std::vector<int>(rows, 0));

    // // init value_var_solution with 0
    // for (int i = 0; i < rows; ++i) {
    //     value_var_solution[i] = new int[cols];
    //     for (int j = 0; j < cols; ++j) {
    //         value_var_solution[i][j] = 0;
    //     }
    // }
    int* diff_var_solution = new int[rows-1]; // store the difference between the value of the variable and the value of the other variables in the solution

    float final_bound = std::numeric_limits<float>::max();
    std::vector<float> bound_test;
    
    // std::vector<int> not_fixed_variables;
    // std::vector<int> fixed_variables;

    std::string dicstr(nb_items,' ');

    // for (int k = 0; k < nb_items; k++){
    //   if (variables[k].size()>=2) {
    //     dicstr[k]='2';
    //     not_fixed_variables.push_back(k);
    //     for (int j=0;j<cols;j++) {
    //             value_var_solution[k][j]=0; }
    //   }

    // else{
    //   fixed_variables.push_back(k);
    //   if (variables[k].val()==1) { 
    //       dicstr[k]='1';
    //         for (int j=0;j<cols;j++) {
    //             value_var_solution[k][j]=variables[k].val(); }
    //     }
    //   else {
    //       dicstr[k]='0';
    //   }
    // }
    // }

    // int size_unfixed=not_fixed_variables.size();
    // int* node_problem=new int[size_unfixed*(nb_constraints+1)+nb_constraints+2 + 1 + 1];

    // TODO : change the subnode writing 

    // node_problem[0]=nb_constraints;
    // node_problem[1]=size_unfixed;

    // for (int k=0;k<size_unfixed;k++) {
    //     node_problem[2+k]=spec.profit(not_fixed_variables[k]);
    //   }

    // for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
    //     int capacity_unfixed=spec.capacity(idx_constraint,nb_items);

    //     for (int i=0;i<fixed_variables.size();i++) {
    //         capacity_unfixed-=spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints)*variables[fixed_variables[i]].val();
    //     }

    //     node_problem[2+size_unfixed+idx_constraint]=capacity_unfixed;
    // }

    // for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
    //     for (int i=0;i<size_unfixed;i++) {
    //         node_problem[2+(idx_constraint+1)*size_unfixed+nb_constraints+i]=spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);
    //     }
    // }

    if (activate_learning_and_adam){
    //   std::vector<torch::jit::IValue> inputs;
    //   // create a tensor with

    //   std::vector<float> nodes;
    //   for (int j=0;j<nb_constraints;j++) {
    //       for (int i=0;i<size_unfixed;i++) {
    //         nodes.push_back(node_problem[2+i]);
    //         nodes.push_back(node_problem[2+ size_unfixed  + nb_constraints + size_unfixed*j +i]);
    //         nodes.push_back((float)std::max((float)node_problem[2+ + size_unfixed + + nb_constraints + size_unfixed*j +i],1.0f) / (float)std::max((float)node_problem[2 + size_unfixed + j], 1.0f));
    //         nodes.push_back((float)node_problem[2 + i] / (float)std::max((float)node_problem[2+ + size_unfixed + nb_constraints +size_unfixed*j +i],1.0f));
    //         nodes.push_back(i);
    //         nodes.push_back(j);
    //       }
    //     }
    //   at::Tensor nodes_t = torch::from_blob(nodes.data(), {nb_constraints * size_unfixed , 6}).to(torch::kFloat32);
    //   inputs.push_back(nodes_t);

    //   std::vector<std::vector<int64_t>> edges_indexes_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
    //   std::vector<std::vector<int64_t>> edges_attr_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
    //   std::vector<float> edges_weights_vec(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed);

    //   int compteur = 0;
    //   for (int k = 0; k < nb_constraints; k++) {
    //     for (int i = 0; i < size_unfixed; i++) {
    //       for (int j = i + 1; j < size_unfixed; j++) {
    //         edges_indexes_vec[0][compteur] = k * size_unfixed + i;
    //         edges_indexes_vec[1][compteur] = k * size_unfixed + j;
    //         edges_weights_vec[compteur] = 1.0f / (float)size_unfixed;
    //         edges_attr_vec[0][compteur] = 0;
    //         edges_attr_vec[1][compteur] = 1;
            
    //         compteur++;
    //       }
    //     }
    //   }

    //   for (int k = 0; k < size_unfixed; k++) {
    //     for (int i = 0; i < nb_constraints; i++) {
    //       edges_indexes_vec[0][compteur] = i * size_unfixed + k;
    //       edges_indexes_vec[1][compteur] = k;
    //       edges_weights_vec[compteur] = 1;
    //       edges_attr_vec[0][compteur] = 1;
    //       edges_attr_vec[1][compteur] = 0;
    //       compteur++;
    //     }
    //   }

    //   // print the elements of edges_indexes_vec.data()
    //   at::Tensor edge_first = torch::from_blob(edges_indexes_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edge_second = torch::from_blob(edges_indexes_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
    //   at::Tensor edges_attr_first = torch::from_blob(edges_attr_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edges_attr_second = torch::from_blob(edges_attr_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edges_attr = torch::cat({edges_attr_first, edges_attr_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
    //   at::Tensor edges_weights = torch::from_blob(edges_weights_vec.data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kFloat32));

    //   inputs.push_back(edges_indexes);
    //   inputs.push_back(edges_weights);
    //   //inputs.push_back(edges_attr.transpose(0, 1));
    //   at::Tensor intermediate_output = module_1.forward(inputs).toTensor();

    //   at::Tensor mean = intermediate_output.mean(0).repeat({nb_constraints * size_unfixed, 1});

    //   std::vector<torch::jit::IValue> intermediate_inputs;

    //   intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));
    //   at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

    //   std::vector<std::vector<float>> multipliers_vec(rows, std::vector<float>(cols, 0.0));
    //   for (int i = 0; i < size_unfixed; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //       multipliers_vec[not_fixed_variables[i]][j] = multipliers[i + j*size_unfixed].item<float>();
    //     }
    //   }
    
    //   std::vector<std::vector<float>> m(rows, std::vector<float>(cols, 0.0));
    //   std::vector<std::vector<float>> v(rows, std::vector<float>(cols, 0.0));
    //   learning_rate = 1.0f;

    //   int k = 1;
    //   while ((( k < 3) || (abs(bound_test[k-2] - bound_test[k-3]) / bound_test[k-2] > 1e-6))&& (k < 300)) { // We repeat the dynamic programming algo to solve the knapsack problem
    //                             // and at each iteration we update the value of the Lagrangian multipliers
    //     final_fixed_bounds = 0.0f;
    //     float bound_iter = 0.0f;
    //     std::vector<SubProblem> subproblems;

    //     for (int i = 0; i < size_unfixed; ++i) {
    //       float sum = 0;
    //       for (int j = 0; j < cols; ++j) {
    //           value_var_solution[not_fixed_variables[i]][j] = 0;
    //       }
    //     }

    //     // we create one subproblem for each knapsack constraint
    //     for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
    //       SubProblem subproblem;
    //       subproblem.weights_sub = new int[nb_items];
    //       subproblem.val_sub = new float[nb_items];
    //       subproblem.capacity = spec.capacity(idx_constraint, nb_items);

    //       for (int i = 0; i < fixed_variables.size(); i++) {
    //         subproblem.weights_sub[i] = spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints);

    //         if (idx_constraint == 0) {
    //           subproblem.val_sub[i] = spec.profit(fixed_variables[i]);
    //         }

    //         else {
    //           subproblem.val_sub[i] = 0.0f;
    //         }
    //       }
    //       for (int i = 0; i < not_fixed_variables.size(); i++) {
    //         subproblem.weights_sub[i + fixed_variables.size()] = spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);

    //         if (idx_constraint == 0) {
    //           float mult = 0.0f;
    //         for (int j = 1; j < nb_constraints; j++) {
    //           mult += multipliers_vec[not_fixed_variables[i]][j];
    //         }
    //           subproblem.val_sub[i + fixed_variables.size()] = spec.profit(not_fixed_variables[i]) +mult;
    //         }

    //         else {
    //           subproblem.val_sub[i + fixed_variables.size()] = -multipliers_vec[not_fixed_variables[i]][idx_constraint];
    //         }
    //       }
    //       subproblem.idx_constraint = idx_constraint;
    //       subproblems.push_back(subproblem);
    //     }

    //     for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
    //       SubProblem subproblem = subproblems[id_subproblem];
    //       float weight_fixed = 0.0f;

    //       for (int k = 0; k < fixed_variables.size(); k++){
    //         weight_fixed+=subproblem.weights_sub[k]* variables[fixed_variables[k]].val();
    //       }

    //       float bound = dp_ssp(subproblem.capacity - weight_fixed,
    //                                 subproblem.weights_sub + fixed_variables.size(),
    //                                 subproblem.val_sub + fixed_variables.size(),
    //                                 nb_items - fixed_variables.size(), nb_constraints,
    //                                 subproblem.idx_constraint,
    //                                 value_var_solution,
    //                                 not_fixed_variables,
    //                                 false);

    //       float fixed_bound = 0.0f;

    //       for (int k = 0; k< fixed_variables.size();k++){
    //           fixed_bound += subproblem.val_sub[k] * variables[fixed_variables[k]].val();
    //       }
    //       final_fixed_bounds += fixed_bound;

    //       bound_iter += bound + fixed_bound; // sum all the bound of the knapsack sub-problem to update the multipliers
      
    //     }
    //     final_bound = std::min(final_bound, bound_iter);
    //     bound_test.push_back(final_bound);


    //     for (int i = 0; i < rows; ++i) {
    //       float sum = 0;
    //       for (int j = 1; j < cols; ++j) {

    //           float gradient = value_var_solution[i][0] - value_var_solution[i][j];

    //           m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradient;

    //           // Update biased second moment estimate
    //           v[i][j] = beta2 * v[i][j] + (1 - beta2) * gradient * gradient;

    //           // Compute bias-corrected first moment estimate
    //           float m_hat = m[i][j] / (1 - std::pow(beta1, k));

    //           // Compute bias-corrected second moment estimate
    //           float v_hat = v[i][j] / (1 - std::pow(beta2, k));

    //         multipliers_vec[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

    //         sum += multipliers_vec[i][j];
    //       }
    //       multipliers_vec[i][0] = sum;
    //     }
    //     // We impose the constraint z <= final_bound
    //     rel(*this, z <= final_bound); 
    //     k++;
    //   }
    // //   std::cout << "final bound : " << final_bound << std::endl;
    // //   std::cout << "k : " << k-1 << std::endl;
    //   compteur_iterations += k-1;

    }
    else if (activate_learning_prediction){
    //   std::vector<torch::jit::IValue> inputs;
    //   // create a tensor with
    //   // at::Tensor nodes = torch::zeros({nb_constraints * size_unfixed, 6});
    //   std::vector<float> nodes;
    //   for (int j=0;j<nb_constraints;j++) {
    //       for (int i=0;i<size_unfixed;i++) {
    //         nodes.push_back(node_problem[2+i]);
    //         nodes.push_back(node_problem[2+ size_unfixed  + nb_constraints + size_unfixed*j +i]);
    //         nodes.push_back((float)std::max((float)node_problem[2+ + size_unfixed + + nb_constraints + size_unfixed*j +i],1.0f) / (float)std::max((float)node_problem[2 + size_unfixed + j], 1.0f));
    //         nodes.push_back((float)node_problem[2 + i] / (float)std::max((float)node_problem[2+ + size_unfixed + nb_constraints +size_unfixed*j +i],1.0f));
    //         nodes.push_back(i);
    //         nodes.push_back(j);
    //       }
    //     }
    //   at::Tensor nodes_t = torch::from_blob(nodes.data(), {nb_constraints * size_unfixed , 6}).to(torch::kFloat32);
    //   inputs.push_back(nodes_t);

    //   std::vector<std::vector<int64_t>> edges_indexes_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
    //   std::vector<std::vector<int64_t>> edges_attr_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
    //   std::vector<float> edges_weights_vec(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed);

    //   int compteur = 0;
    //   for (int k = 0; k < nb_constraints; k++) {
    //     for (int i = 0; i < size_unfixed; i++) {
    //       for (int j = i + 1; j < size_unfixed; j++) {
    //         edges_indexes_vec[0][compteur] = k * size_unfixed + i;
    //         edges_indexes_vec[1][compteur] = k * size_unfixed + j;
    //         edges_weights_vec[compteur] = 1.0f / (float)size_unfixed;
    //         edges_attr_vec[0][compteur] = 0;
    //         edges_attr_vec[1][compteur] = 1;
            
    //         compteur++;
    //       }
    //     }
    //   }
    //   for (int k = 0; k < size_unfixed; k++) {
    //     for (int i = 0; i < nb_constraints; i++) {
    //       edges_indexes_vec[0][compteur] = i * size_unfixed + k;
    //       edges_indexes_vec[1][compteur] = k;
    //       edges_weights_vec[compteur] = 1;
    //       edges_attr_vec[0][compteur] = 1;
    //       edges_attr_vec[1][compteur] = 0;
    //       compteur++;
    //     }
    //   }
    //   // print the elements of edges_indexes_vec.data()
    //   at::Tensor edge_first = torch::from_blob(edges_indexes_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edge_second = torch::from_blob(edges_indexes_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
    //   at::Tensor edges_attr_first = torch::from_blob(edges_attr_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edges_attr_second = torch::from_blob(edges_attr_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
    //   at::Tensor edges_attr = torch::cat({edges_attr_first, edges_attr_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
    //   at::Tensor edges_weights = torch::from_blob(edges_weights_vec.data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kFloat32));

    //   inputs.push_back(edges_indexes);
    //   inputs.push_back(edges_weights);
    //   //inputs.push_back(edges_attr.transpose(0, 1));
    //   at::Tensor intermediate_output = module_1.forward(inputs).toTensor();


    //   at::Tensor mean = intermediate_output.mean(0).repeat({nb_constraints * size_unfixed, 1});

    //   std::vector<torch::jit::IValue> intermediate_inputs;

    //   intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));
    //   at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

    //   // create a vector of multipliers
    //   std::vector<float> multipliers_vec;
    //   multipliers_vec.resize( nb_constraints * size_unfixed);
    //   for (int i = 0; i < nb_constraints * size_unfixed; i++) {
    //     multipliers_vec[i] = multipliers[i].item<float>();
    //   }

    //   float final_bound = 0.0f;
    //   std::vector<SubProblem> subproblems;

    //     // we create one subproblem for each knapsack constraint
    //   for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
    //     SubProblem subproblem;
    //     subproblem.weights_sub = new int[nb_items];
    //     subproblem.val_sub = new float[nb_items];
    //     subproblem.capacity = spec.capacity(idx_constraint, nb_items);

    //     for (int i = 0; i < fixed_variables.size(); i++) {
    //       subproblem.weights_sub[i] = spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints);

    //       if (idx_constraint == 0) {
    //         subproblem.val_sub[i] = spec.profit(fixed_variables[i]);
    //       }

    //       else {
    //         subproblem.val_sub[i] = 0.0f;
    //       }
    //     }
    //     for (int i = 0; i < not_fixed_variables.size(); i++) {
    //       subproblem.weights_sub[i + fixed_variables.size()] = spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);

    //       if (idx_constraint == 0) {
    //         float mult = 0.0f;
    //         for (int j = 1; j < nb_constraints; j++) {
    //           mult += multipliers_vec[i + size_unfixed * j];
    //         }
    //         subproblem.val_sub[i + fixed_variables.size()] = spec.profit(not_fixed_variables[i]) + mult ;
    //       }

    //       else {
    //         subproblem.val_sub[i + fixed_variables.size()] = - multipliers_vec[i + size_unfixed * idx_constraint];
    //       }
    //     }
    //     subproblem.idx_constraint = idx_constraint;
    //     subproblems.push_back(subproblem);
    //   }

    //   for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
    //     SubProblem subproblem = subproblems[id_subproblem];
    //     float weight_fixed = 0.0f;
    
    //     for (int k = 0; k < fixed_variables.size(); k++){
    //       weight_fixed+=subproblem.weights_sub[k]* variables[fixed_variables[k]].val();
    //     }

    //     float bound = dp_ssp(subproblem.capacity - weight_fixed,
    //                                 subproblem.weights_sub + fixed_variables.size(),
    //                                 subproblem.val_sub + fixed_variables.size(),
    //                                 nb_items - fixed_variables.size(), nb_constraints,
    //                                 subproblem.idx_constraint,
    //                                 value_var_solution,
    //                                 not_fixed_variables,
    //                                 false);

    //     float fixed_bound = 0.0f;

    //     for (int k = 0; k< fixed_variables.size();k++){
    //         fixed_bound += subproblem.val_sub[k] * variables[fixed_variables[k]].val();
    //     }
    //     final_fixed_bounds += fixed_bound;
    //     final_bound += bound + fixed_bound; // sum all the bound of the knapsack sub-problem to update the multipliers
      
    //   }
    //   // We impose the constraint z <= final_bound
    //   rel(*this, z <= final_bound); 
    // //   std::cout << "final bound : " << final_bound << std::endl;
   
    }
    else if (activate_adam){
    //   std::vector<std::vector<float>> m(rows, std::vector<float>(cols, 0.0));
    //   std::vector<std::vector<float>> v(rows, std::vector<float>(cols, 0.0));
    //   learning_rate = 1.0f;

    //   int k = 1;
    //   while ((( k < 5) || (abs(bound_test[k-2] - bound_test[k-3]) / bound_test[k-2] > 1e-6)) && (k < 10)) { 
    //     // We repeat the dynamic programming algo to solve the knapsack problem
    //     // and at each iteration we update the value of the Lagrangian multipliers
    //     final_fixed_bounds = 0.0f;
    //     float bound_iter = 0.0f;
    //     std::vector<SubProblem> subproblems;

    //     // we create one subproblem for each knapsack constraint
    //     for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
    //       SubProblem subproblem;
    //       subproblem.val_sub = std::vector(nb_items, std::vector<float>());
    //       subproblem.domains_sub = std::vector(nb_items, std::vector<int>());
    //       subproblem.states = std::vector(nb_states, 0);
    //       subproblem.transitions_sub = std::vector<std::tuple<int, int, int>>();

    //       for (int i = 0; i < nb_items; i++) {
    //         for (IntVarValues j(variables[i]); j(); ++j) {

    //           subproblem.domains_sub[i].push_back(j.val());
    //         }
    //       }

    //       for (int i = 0; i < nb_items; i++){
    //         for (int j = 0; j < nb_states; j++) {
    //           if (idx_constraint == 0) {
    //             subproblem.val_sub[i].push_back(spec.profit(i, j, F) + multipliers[i][idx_constraint][j]);
    //           }
    //           else {
    //             subproblem.val_sub[i].push_back(- multipliers[i][idx_constraint][j]);
    //           }
    //         }
    //       }

    //       for (int i = 0; i < nb_states; i++) {
    //         for (IntVarValues k(variables[i]); k(); ++k) {
    //           for (int j = 0; j < nb_states; j++) {
    //             if (spec.transition(idx_constraint, i, k.val(), j, F, nb_items, nb_constraints,nb_states, nb_values))
    //             subproblem.transitions_sub.push_back(std::tuple(i, k.val(), j));
    //           }
    //         }
    //       }

    //       subproblem.idx_constraint = idx_constraint;
    //       subproblems.push_back(subproblem);
    //     }

    //     for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
    //       SubProblem subproblem = subproblems[id_subproblem];

    //       float bound = dp_ssp(subproblem.val_sub,
    //                             subproblem.domains_sub,
    //                             subproblem.transitions_sub,
    //                             subproblem.states, 
    //                             nb_states, 
    //                             nb_values, 
    //                             nb_items, 
    //                             initial_state, 
    //                             value_var_solution[id_subproblem], 
    //                             false);

    //        bound_iter += bound; // sum all the bound of the knapsack sub-problem to update the multipliers
    //     //   std::cout << "bound " << bound << std::endl;
    //     //   for (int i = 0; i < rows; ++i) {
    //     //     std::cout << multipliers[i][id_subproblem] << " ";
    //     //   std::cout << std::endl;
    //     // }
    //     //   for (int i = 0; i < nb_items; i++) {
    //     //       std::cout << value_var_solution[id_subproblem][i] << " ";
    //     //     std::cout << std::endl;
    //     //   }
    //     }
    //     final_bound = std::min(final_bound, bound_iter);
    //     bound_test.push_back(bound_iter);

    //     for (int i = 0; i < rows; ++i) {
    //       float sum = 0;
    //       for (int j = 1; j < cols; ++j) {

    //           float gradient = value_var_solution[0][i] - value_var_solution[j][i];

    //           m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradient;

    //           // Update biased second moment estimate
    //           v[i][j] = beta2 * v[i][j] + (1 - beta2) * gradient * gradient;

    //           // Compute bias-corrected first moment estimate
    //           float m_hat = m[i][j] / (1 - std::pow(beta1, k));

    //           // Compute bias-corrected second moment estimate
    //           float v_hat = v[i][j] / (1 - std::pow(beta2, k));

    //         multipliers[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

    //         sum += multipliers[i][j];
    //       }
    //       multipliers[i][0] = sum;
    //     }
    //     // We impose the constraint z <= final_bound
    //     rel(*this, z <= final_bound); 
    //     k++;
    //   }
    // //   std::cout << "final bound : " << final_bound << std::endl;
    // //   std::cout << "k : " << k-1 << std::endl;
    //   compteur_iterations+=k-1;
    }
    else{
      int nb_iter = 1;
      while ((( nb_iter < 5) || (abs(bound_test[nb_iter-2] - bound_test[nb_iter-3]) / bound_test[nb_iter-2] > 1e-6)) && (nb_iter< 50)) { 
        // We repeat the dynamic programming algo to solve the knapsack problem
        // and at each iteration we update the value of the Lagrangian multipliers
        final_fixed_bounds = 0.0f;
        float bound_iter = 0.0f;
        std::vector<SubProblem> subproblems;

        // we create one subproblem for each knapsack constraint
        for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
          SubProblem subproblem;
          subproblem.val_sub = std::vector(nb_items, std::vector<float>());
          subproblem.domains_sub = std::vector(nb_items, std::vector<int>());
          subproblem.states = std::vector(nb_states, 0);
          subproblem.final_states = std::vector(F, 0);
          subproblem.transitions_sub = std::vector<std::tuple<int, int, int>>();

          for (int i = 0; i < nb_items; i++) {
            for (IntVarValues j(variables[i]); j(); ++j) {
              subproblem.domains_sub[i].push_back(j.val());
            }
          }

          for (int i = 0; i < nb_states; i++) {
            subproblem.states[i] = i;
          }

          for (int i = 0; i < F; i++) {
            subproblem.final_states.push_back(spec.final_state(i, F));
          }

          for (int i = 0; i < nb_items; i++){
            for (int j = 0; j < nb_values; j++) {
              if (idx_constraint == 0) {
                subproblem.val_sub[i].push_back(spec.profit(i, j, F) + multipliers[i][idx_constraint][j]);
              }
              else {
                subproblem.val_sub[i].push_back(-multipliers[i][idx_constraint][j]);
              }
            }
          }

          for (int i = 0; i < nb_states; i++) {
            for (int k = 0; k < nb_values; k++) {
              for (int j = 0; j < nb_states; j++) {
                if (spec.transition(idx_constraint, i, k, j, F, nb_items, nb_constraints,nb_states, nb_values))
                subproblem.transitions_sub.push_back(std::tuple(i, k, j));
              }
            }
          }


          subproblem.idx_constraint = idx_constraint;
          subproblems.push_back(subproblem);
        }

        for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
          SubProblem subproblem = subproblems[id_subproblem];

          float bound = dp_ssp(subproblem.val_sub,
                                subproblem.domains_sub,
                                subproblem.transitions_sub,
                                subproblem.states, 
                                subproblem.final_states,
                                nb_states, 
                                nb_values, 
                                nb_items, 
                                initial_state, 
                                value_var_solution[id_subproblem], 
                                false);

           bound_iter += bound; // sum all the bound of the knapsack sub-problem to update the multipliers
          std::cout << "bound " << bound << std::endl;
        //   for (int i = 0; i < rows; ++i) {
        //     std::cout << multipliers[i][id_subproblem] << " ";
        //   std::cout << std::endl;
        // }
          for (int i = 0; i < nb_items; i++) {
              std::cout << value_var_solution[id_subproblem][i] << " ";
          }
          std::cout << std::endl;
        }
        final_bound = std::min(final_bound, bound_iter);
        bound_test.push_back(bound_iter);

        for (int i = 0; i < rows; ++i) {
          for (int j = 1; j < cols; ++j) {

            multipliers[i][j][value_var_solution[j][i]] = multipliers[i][j][value_var_solution[j][i]] + learning_rate;

            multipliers[i][j][value_var_solution[0][i]] = multipliers[i][j][value_var_solution[0][i]] - learning_rate;
          }
        }

        for (int i = 0; i < rows; ++i) {
          for (int k = 0; k < nb_values; ++k) {
            float sum = 0;
          for (int j = 1; j < cols; ++j) {
            sum += multipliers[i][j][k];
          }
          multipliers[i][0][k] = sum;
          }
        }

        if ( nb_iter %1 ==0) std::cout << "Iteration " << nb_iter << " : " << bound_iter << std::endl;
        // We impose the constraint z <= final_bound
        //std::cout << "final bound : " << final_bound << std::endl;
        nb_iter++;
      }
      rel(*this, z <= final_bound); 
      //std::cout << "final bound : " << final_bound << std::endl;
    }

    // node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+2] = final_fixed_bounds;
    // node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+2 + 1] = final_bound;
    // if (write_samples) {
    //   if ((set_nodes.count(dicstr)==0) and (size_unfixed>=5) ) { // and (size_unfixed>=5)  and (size_unfixed<=30)
    //       set_nodes.insert(dicstr);

    //       if (outputFileMK->is_open()) {
    //       for (int i=0;i<size_unfixed*(nb_constraints+1)+nb_constraints+3;i++) {
    //           *outputFileMK << node_problem[i]<<",";
    //         }

    //       *outputFileMK << node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+3]<<"\n";
    //       } 
    //     } 
    //   }

    // for (int i = 0; i < rows; ++i) {
    //   delete[] value_var_solution[i];
    // }

    // delete[] value_var_solution;
    }

    static void post(Space& home) {
        static_cast<SSP&>(home).more();
    }

  /// Return cost
    virtual IntVar cost(void) const {
        return z;
    }
  /// Constructor for cloning \a s
    SSP(SSP& s)
        : IntMaximizeSpace(s), spec(s.spec) {
        variables.update(*this, s.variables);
        z.update(*this, s.z);
        this->order_branching = s.order_branching;
        this->activate_bound_computation = s.activate_bound_computation;
        this->activate_adam = s.activate_adam;
        this->activate_learning_prediction = s.activate_learning_prediction;
        this->activate_learning_and_adam = s.activate_learning_and_adam;
        this->activate_heuristic = s.activate_heuristic;
        this->K = s.K;
        this->learning_rate = s.learning_rate;
        this->init_value_multipliers = s.init_value_multipliers;
        this->multipliers = s.multipliers;
        this->outputFileMK=s.outputFileMK;
        this->write_samples=s.write_samples;
    }
  /// Copy during cloning
  virtual Space*
    copy(void) {
        return new SSP(*this);
    }

    virtual void constrain(const Space& _b) { // compute the bound at each leaf node giving a solution
        const SSP& b = static_cast<const SSP&>(_b);
        std::cout<<"solution: "<<b.z<<std::endl;
        // We impose the constraint z >= current sol
        rel(*this, z >= b.z);
}

// Define a structure for the nodes
struct Node {
    int layer;
    int state;

    bool operator==(const Node& other) const {
        return layer == other.layer && state == other.state;
    }
};

// Define a structure for the edges
struct Edge {
    Node from;
    Node to;
    int label;
    double cost;
};

// Hash function for the Node to use in unordered_map
struct NodeHash {
    size_t operator()(const Node& node) const {
        return std::hash<int>()(node.layer) ^ std::hash<int>()(node.state);
    }
};

//TODO: change the routine to solve subproblems with Pesant's filtering algorithm and DP
float dp_ssp(std::vector<std::vector<float>> profits,
              std::vector<std::vector<int>> domain_values,
              std::vector<std::tuple<int, int, int>> transitions,
              std::vector<int> states,
              std::vector<int> final_states,
              int nb_states,
              int nb_values,
              int nb_items,
              int initialState,
              std::vector<int>& value_var_solution,
                  bool verbose=false) {

    // Build a graph
    
    std::vector<std::unordered_map<int, Node>> layers(nb_items + 1);
    std::unordered_map<int, std::vector<Edge>> adjList;

    layers[0][0] = {0,initialState};

    for (int j = 1; j <= nb_items -1; ++j) {
        for (const auto& [key, prevNode] : layers[j - 1]){
            for (int a : domain_values[j - 1]) {
                for (const auto& [q1, label, q2] : transitions) {
                    if (prevNode.state == q1 && label == a) {
                        Node newNode = {j, q2};
                        layers[j][newNode.state] = newNode;
                        double cost = profits[j - 1][a];
                        adjList[j-1].push_back({prevNode, newNode, a, cost});
                    }
                }
            }
        }
    }

    for (const auto& [key, prevNode] : layers[nb_items - 1]){
            for (int a : domain_values[nb_items - 1]) {
                for (const auto& [q1, label, q2] : transitions) {
                    if (prevNode.state == q1 && label == a && std::find(final_states.begin(), final_states.end(), q2) != final_states.end()){
                        Node newNode = {nb_items, q2};
                        layers[nb_items][newNode.state] = newNode;
                        double cost = profits[nb_items - 1][a];
                        adjList[nb_items-1].push_back({prevNode, newNode, a, cost});
                    }
                }
            }
        }


    std::unordered_map<Node, double, NodeHash> R;
    std::unordered_map<Node, std::pair<Node, int>, NodeHash> predecessor; // To store the path

    R[{0, initialState}] = 0;

    for (int j = 1; j <= nb_items; ++j) {
        for (const auto& [key, node] : layers[j]){
            double maxCost = std::numeric_limits<double>::lowest();
            Node bestPredecessor;
            int bestLabel = - 1;
            for (const Edge& edge : adjList[j - 1]) {
                if (edge.to.layer == j && edge.to.state == node.state) {
                    double newCost = R[{j - 1, edge.from.state}] + edge.cost;
                    if (newCost > maxCost) {
                        maxCost = newCost;
                        bestPredecessor = edge.from;
                        bestLabel = edge.label;
                    }
                }
            }
            R[node] = maxCost;
            if (maxCost != std::numeric_limits<double>::lowest()) {
                predecessor[node] = {bestPredecessor, bestLabel};
            }
        }
    }

    // Find the optimal bound for the constraint
    double optimalBound = std::numeric_limits<double>::lowest();
    Node endNode;
    for (const int q : states) {
        Node finalNode = {nb_items, q};
        if (R[finalNode] > optimalBound) {
            optimalBound = R[finalNode];
            endNode = finalNode;
        }
    }

    // Backtrack to find the solution path
    std::vector<int> solutionPath;
    Node currentNode = endNode;
    while (currentNode.layer > 0) {
        solutionPath.push_back(predecessor[currentNode].second);
        currentNode = predecessor[currentNode].first;
    }
    reverse(solutionPath.begin(), solutionPath.end());

    value_var_solution = solutionPath; 

    return optimalBound;
}

  /// Print solution
  virtual void
  print(std::ostream& os) const {
    os << "z: " << z << std::endl;
    os << "variables: " << variables << std::endl;
  }
};

int main(int argc, char* argv[]) {

    std::string n_size = argv[1];
    std::string n_model = argv[2];
    int number_of_sizes = std::stoi(n_size);
    int number_of_models = std::stoi(n_model);

    std::string sizes[] = {"30", "50", "100", "200"};

    bool activate_bound_computation[] = {true, true, true, false};
    bool activate_adam[] = {true, false, false, false};
    bool activate_learning_prediction[] = {false, true, false, false};
    bool activate_learning_and_adam[] = {false, false, true, false};
    bool activate_heuristic = true;
    int K = 500;
    float learning_rate = 1.0f;
    float init_value_multipliers = 1.0f;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // module_1 = torch::jit::load("/home/darius/scratch/learning-bounds/trained_models/mknapsack/model_graph_representation.pt");
        // module_2 = torch::jit::load("/home/darius/scratch/learning-bounds/trained_models/mknapsack/model_prediction.pt");

    }
    catch (const c10::Error& e) {
        std::cerr << "error with loading the models \n";
        return -1;
    }

        bool write_samples;
    if (strcmp(argv[3], "write_samples") == 0)
        write_samples = true;
    else
        write_samples = false;

        if (write_samples) {
            for (int index_size = 2; index_size < number_of_sizes; index_size++) {
            
            std::ifstream inputFilea("/Users/dariusdabert/Documents/Documents/X/3A/Stage Polytechnique Montréal/learning-bounds/data/ssp/train/ssp-data-trainset" + sizes[index_size] + ".txt");

            std::ofstream outputFilea("/Users/dariusdabert/Documents/Documents/X/3A/Stage Polytechnique Montréal/learning-bounds/data/ssp/train/trainset-"+ sizes[index_size] + "-subnodes.txt");
            bool activate_bound_computation = true;
            bool activate_adam = true;
            bool activate_learning_prediction = false;
            bool activate_learning_and_adam = false;

        std::string line;
        std::vector<int> problem;
        std::vector<int> numbers;
        int j = 1;

        while (std::getline(inputFilea, line)) {

            //std::ofstream outputFilea(std::to_string(j) + ".txt");

            set_nodes.clear();
            std::vector<int> problem;
            std::istringstream iss(line);
            std::string substring;
            while (std::getline(iss, substring, ',')) {
                    problem.push_back(std::stoi(substring));
                }
            std::cout<<""<<std::endl;
            OptionsSSP opt=OptionsSSP(activate_bound_computation, activate_adam, activate_learning_prediction, activate_learning_and_adam, activate_heuristic, K,learning_rate,init_value_multipliers, &outputFilea , problem, true);
            opt.instance();
            opt.solutions(0);
            opt.parse(argc,argv);
            IntMaximizeScript::run<SSP,BAB,OptionsSSP>(opt);
            j++;
            }
            outputFilea.close();
            inputFilea.close(); // Close the file when done
            }

        }
        else {
    for (int index_size = 0; index_size < 1; index_size++) {
        for (int index_model = 0; index_model < 1; index_model++ ){

            for (int i = 0; i < 1; i++) {
            std::ifstream inputFilea("/Users/dariusdabert/Documents/Documents/X/3A/Stage Polytechnique Montréal/learning-bounds/data/ssp/train/ssp-data-trainset10-20.txt");
            std::string line;
            std::vector<int> problem;
            std::vector<int> numbers;
            int j =1;

            while (std::getline(inputFilea, line)) {
                compteur_iterations = 0;
                std::vector<int> problem;
                std::istringstream iss(line);
                std::string substring;
                    while (std::getline(iss, substring, ',')) {
                        problem.push_back(std::stoi(substring));
                    }
                std::cout<<""<<std::endl;
                OptionsSSP opt=OptionsSSP(false, false, false, false, false, K,learning_rate,init_value_multipliers, NULL , problem, false);
                opt.instance();
                opt.solutions(0);
                opt.parse(argc,argv);
                IntMaximizeScript::run<SSP,BAB,OptionsSSP>(opt);
                std::cout << "compteur_iterations : " << compteur_iterations << std::endl;
                }
                j++;
        
        std::cout<<"separateur_de_modeles"<<std::endl;
        inputFilea.close(); // Close the file when done
            }
        }
        std::cout << "separateur de taille" << std::endl;
    }
        }
    return 0;
    }

namespace {
    int* mknps[] = {
    };

    const char* name[] = {
        "mknapsacks",
        "n1c1w1_b",
        "n1c1w1_c",
        "n1c1w1_d",
        "n1c1w1_e",
        "n1c1w1_f",
        "n1c1w1_g",
        "n1c1w1_h",
        "n1c1w1_i",
        "n1c1w1_j",
        "n1c1w1_k",
        "n1c1w1_l",
        "n1c1w1_m",
        "n1c1w1_n",
        "n1c1w1_o",
        "n1c1w1_p",
        "n1c1w1_q",
        "n1c1w1_r",
        "n1c1w1_s",
        "n1c1w1_t",
        "n1c1w1_u",
        "n1c1w1_v",
        "n1c1w1_w",
        "n1c1w1_x",
        "n1c1w1_y",
        "n1c1w1_z",
        "n1c1w1_a_1",
        "n1c1w1_b_1",
        "n1c1w1_c_1",
        "n1c1w1_d_1",
        "n1c1w1_e_1"
    };

    }
