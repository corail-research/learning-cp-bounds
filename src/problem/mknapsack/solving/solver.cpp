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

// Allow to store the multipliers in a local handle to share them between the nodes
class LI : public LocalHandle {
protected:
  class LIO : public LocalObject {
  public:
    std::vector<std::vector<float>> multipliers;
    LIO(Space& home, std::vector<std::vector<float>> d) : LocalObject(home), multipliers(d) {}
    LIO(Space& home, LIO& l)
      : LocalObject(home,l), multipliers(l.multipliers) {}
    virtual LocalObject* copy(Space& home) {
      return new (home) LIO(home,*this);
    }
    virtual size_t dispose(Space&) { return sizeof(*this); }
  };

public:
  LI(Space& home, std::vector<std::vector<float>> d)
    : LocalHandle(new (home) LIO(home,d)) {}
  LI(const LI& li)
    : LocalHandle(li) {}
  LI& operator =(const LI& li) {
    return static_cast<LI&>(LocalHandle::operator =(li));
  }
  std::vector<std::vector<float>> get(void) const {
    return static_cast<LIO*>(object())->multipliers;
  }
  void set(std::vector<std::vector<float>> d) {
    static_cast<LIO*>(object())->multipliers = d;
  }
};

// Structure of the subproblems of the knapsack problem
struct SubProblem {
  int* weights_sub;
  float* val_sub;
  int capacity;
  int idx_constraint;
};

// Instance data
namespace {
  // Instances
  extern int* mknps[];
  // Instance names
  extern const char* name[];

  /// A wrapper class for instance data
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
    /// Return the profit of item i (index of item starting at 0)
    int profit(int i) const {
      return pData[2+i];
    }
    /// Return the capacity of the knapsack i
    int capacity(int i, int nb_items) const {
      return pData[2+nb_items+i];
    }
    /// Return the weights of the item i in knapsack j (i and j starting at 0)
    int weight(int j, int i, int nb_items, int nb_constraints) const {
      return pData[2+nb_items+nb_constraints+j*nb_items+i];
    }
    /// Return the optimal value (last value in the instance file)
    int optimal(int nb_items, int nb_constraints) const {
      return pData[2+nb_items+nb_constraints+nb_constraints*nb_items+1];
    }
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
        return(2+nb_items()+nb_constraints()+nb_items()*nb_constraints());
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
class OptionsKnapsack: public InstanceOptions {
public:
  bool activate_bound_computation;
  bool activate_learning_prediction;
  bool activate_heuristic;
  int K;
  float learning_rate;
  float init_value_multipliers;
  std::ofstream* outputFile;
  bool write_samples;
  const char* s;
  std::vector<int> problem={};
    OptionsKnapsack(bool activate_bound_computation0,
    bool activate_learning_prediction0,
    bool activate_heuristic0, 
    int K0, float learning_rate0, 
    float init_value_multipliers0, 
    std::ofstream* outputFile_0, 
    std::vector<int> problem0,
    bool write_samples0=true,const char* s0="")
    : InstanceOptions("MultiKnapsack"), s(s0), 
    activate_bound_computation(activate_bound_computation0),
    activate_learning_prediction(activate_learning_prediction0),  
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

class MultiKnapsack : public IntMaximizeSpace {
protected:
    const Spec spec; // Specification of the instance
    BoolVarArray variables; // Decision variables for each item
    IntVar z; // Variable for the objective function
    bool activate_bound_computation;
    bool activate_learning_prediction;
    bool activate_heuristic;
    int K;
    float learning_rate;
    float init_value_multipliers;
    std::vector<int> order_branching; // Order of the items to branch on
    std::vector<std::vector<float>> multipliers; // Lagrangian multipliers shared between the nodes
    std::ofstream* outputFileMK;
    bool write_samples;
public:

  class NoneMax : public Brancher {
  protected:
  ViewArray<Int::BoolView> variables;
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
  NoneMax(Home home, ViewArray<Int::BoolView>& variables0)
    : Brancher(home), variables(variables0) {}

  static void post(Home home, ViewArray<Int::BoolView>& variables) {
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
  virtual Choice* choice(Space& home) {
    bool activate_heuristic=static_cast<MultiKnapsack&>(home).activate_heuristic;
    for (int i=0; true; i++){
        int index;
    if (activate_heuristic) {
        index = static_cast<MultiKnapsack&>(home).order_branching[i]; }
    else {
        index=i;
    }
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
    bool activate_bound=static_cast<MultiKnapsack&>(home).activate_bound_computation;
    const PosVal& pv = static_cast<const PosVal&>(c);
    int pos=pv.pos, val=pv.val;
    if (a == 0){
      ExecStatus temp = me_failed(variables[pos].eq(home,val)) ? ES_FAILED : ES_OK;
      if (activate_bound)
         static_cast<MultiKnapsack&>(home).more();
      return temp;
    }
    else{
      ExecStatus temp = me_failed(variables[pos].nq(home,val)) ? ES_FAILED : ES_OK;
      if (activate_bound)
        static_cast<MultiKnapsack&>(home).more();
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

void nonemax(Home home, const BoolVarArgs& variables) {
  if (home.failed()) return;
  ViewArray<Int::BoolView> y(home,variables);
  NoneMax::post(home,y);
}

  /// Actual model
  MultiKnapsack(const OptionsKnapsack& opt)
  : IntMaximizeSpace(),
    spec(opt.problem,opt.s),
    variables(*this, spec.nb_items(), 0, 1),
    z(*this, spec.lower(), spec.upper()),
    outputFileMK(opt.outputFile),
    write_samples(opt.write_samples){
      int n = spec.nb_items();        // The number of items
      int m = spec.nb_constraints();  // The number of constraints
      int profits[n];                 // The profit of the items
      int capacities[m];              // The capacities of the knapsacks
      int weights[m][n];              // The weights of the items in the knapsacks (one vector per knapsack)
      this->activate_bound_computation = opt.activate_bound_computation; // Activate the bound computation at each node
      this->activate_learning_prediction = opt.activate_learning_prediction; // Activate the learning prediction
      this->activate_heuristic = opt.activate_heuristic; // Activate the heuristic to branch on the items
      this->K = opt.K;                // The number of iteration to find the optimal multipliers
      this->learning_rate = opt.learning_rate; // The learning rate to update the multipliers
      this->init_value_multipliers = opt.init_value_multipliers; // The starting value of the multipliers
      
      std::vector<std::vector<float>> v; // help to initialize the local handle which will contain the multipliers and be shared between the nodes
      v.resize(n);
      for (int i = 0; i < n; ++i) {
          float sum = 0;
          v[i].resize(m);
          for (int j = 1; j < m; ++j) {
              v[i][j] = - spec.profit(i) * 0.05f ;
              sum += - spec.profit(i) * 0.05f; 
          }
          v[i][0] = sum;
      }

      LI multipliers(*this, v);
      this->multipliers = multipliers.get();

      // Get the profits, capacities and weights of the items from the instance
      for (int i = 0; i < n; i++) {
          profits[i] = spec.profit(i);
      }

      for (int i = 0; i < m; i++) {
          capacities[i] = spec.capacity(i, n);
      }

      for (int j = 0; j < m; j++) {
          for (int i = 0; i < n; i++) {
              int w = spec.weight(j, i, n, m);
              weights[j][i] = w;
          }
      }

      this->order_branching.resize(n);

      // order is the list of the index of items sorted by decreasing ratio between profit and weight
      for (int i = 0; i < n; i++) {
        order_branching[i] = i;
      }

      std::sort(order_branching.begin(), order_branching.end(), [&](int i, int j) { 
        float sum_i = 0;
        float sum_j = 0;
        for (int k = 0; k < m; k++) {
          sum_i += weights[k][i];
          sum_j += weights[k][j];
        }
        float ratio_i = profits[i] / sum_i;
        float ratio_j = profits[j] / sum_j;
        return ratio_i > ratio_j;
      });

      // The objective function
      IntVarArgs profits_x;
      for (int i = 0; i < n; i++) {
        profits_x << expr(*this, profits[i] * variables[i]);
      }
      z = expr(*this, sum(profits_x));

      // The constraints for the knapsacks
      for (int j = 0; j < m; j++) {
          IntVarArgs weight_x;
          for (int i = 0; i < n; i++) {
              weight_x << expr(*this, weights[j][i] * variables[i]);
          }
          linear(*this, weight_x, IRT_LQ, capacities[j]);
      }

    nonemax(*this, variables);
  }

  void compare(const Space& s, std::ostream& os) const {
}

  void more(void) { // compute the bound at each node after every branching
    int nb_items = spec.nb_items();
    int nb_constraints = spec.nb_constraints();
    int capacity = spec.capacity(0, nb_items);
    int weights[nb_items];
    int val[nb_items];
    float final_fixed_bounds = 0.0f;
    for (int i = 0; i < nb_items; i++) {
      weights[i] = spec.weight(0, i, nb_items, nb_constraints);
      val[i] = spec.profit(i);
    }

    int rows = nb_items;
    int cols = nb_constraints;

    // store the value of the variable in the solution during the dynamic programming algo to update the multipliers
    int** value_var_solution = new int*[rows];

    // init value_var_solution with 0
    for (int i = 0; i < rows; ++i) {
        value_var_solution[i] = new int[cols];
        for (int j = 0; j < cols; ++j) {
            value_var_solution[i][j] = 0;
        }
    }
    int* diff_var_solution =new int[rows-1]; // store the difference between the value of the variable and the value of the other variables in the solution

    float final_bound = std::numeric_limits<float>::max();
    float bound_test[K];
    
    std::vector<int> not_fixed_variables;
    std::vector<int> fixed_variables;

    std::string dicstr(nb_items,' ');

    for (int k = 0; k < nb_items; k++){
      if (variables[k].size()==2) {
        dicstr[k]='2';
        not_fixed_variables.push_back(k);
        for (int j=0;j<cols;j++) {
                value_var_solution[k][j]=0; }
      }

    else{
      fixed_variables.push_back(k);
      if (variables[k].val()==1) { 
          dicstr[k]='1';
            for (int j=0;j<cols;j++) {
                value_var_solution[k][j]=1; }
        }
      else {
          dicstr[k]='0';
      }
    }
    }

    int size_unfixed=not_fixed_variables.size();
    int* node_problem=new int[size_unfixed*(nb_constraints+1)+nb_constraints+2 + 1 + 1];

    node_problem[0]=nb_constraints;
    node_problem[1]=size_unfixed;

    for (int k=0;k<size_unfixed;k++) {
        node_problem[2+k]=spec.profit(not_fixed_variables[k]);
      }

    for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
        int capacity_unfixed=spec.capacity(idx_constraint,nb_items);

        for (int i=0;i<fixed_variables.size();i++) {
            capacity_unfixed-=spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints)*variables[fixed_variables[i]].val();
        }

        node_problem[2+size_unfixed+idx_constraint]=capacity_unfixed;
    }

    for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
        for (int i=0;i<size_unfixed;i++) {
            node_problem[2+(idx_constraint+1)*size_unfixed+nb_constraints+i]=spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);
        }
    }

    if (activate_learning_prediction){
      std::vector<torch::jit::IValue> inputs;
      // create a tensor with
      // at::Tensor nodes = torch::zeros({nb_constraints * size_unfixed, 6});
      std::vector<float> nodes;
      for (int j=0;j<nb_constraints;j++) {
          for (int i=0;i<size_unfixed;i++) {
            nodes.push_back(node_problem[2+i]);
            nodes.push_back(node_problem[2+ size_unfixed  + nb_constraints + size_unfixed*j +i]);
            nodes.push_back((float)std::max((float)node_problem[2+ + size_unfixed + + nb_constraints + size_unfixed*j +i],1.0f) / (float)std::max((float)node_problem[2 + size_unfixed + j], 1.0f));
            nodes.push_back((float)node_problem[2 + i] / (float)std::max((float)node_problem[2+ + size_unfixed + nb_constraints +size_unfixed*j +i],1.0f));
            nodes.push_back(i);
            nodes.push_back(j);
          }
        }
      at::Tensor nodes_t = torch::from_blob(nodes.data(), {nb_constraints * size_unfixed , 6}).to(torch::kFloat32);
      inputs.push_back(nodes_t);

      std::vector<std::vector<int64_t>> edges_indexes_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
      std::vector<std::vector<int64_t>> edges_attr_vec(2, std::vector<int64_t>(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed));
      std::vector<float> edges_weights_vec(nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed);

      int compteur = 0;
      for (int k = 0; k < nb_constraints; k++) {
        for (int i = 0; i < size_unfixed; i++) {
          for (int j = i + 1; j < size_unfixed; j++) {
            edges_indexes_vec[0][compteur] = k * size_unfixed + i;
            edges_indexes_vec[1][compteur] = k * size_unfixed + j;
            edges_weights_vec[compteur] = 1.0f / (float)size_unfixed;
            edges_attr_vec[0][compteur] = 0;
            edges_attr_vec[1][compteur] = 1;
            
            compteur++;
          }
        }
      }
      for (int k = 0; k < size_unfixed; k++) {
        for (int i = 0; i < nb_constraints; i++) {
          edges_indexes_vec[0][compteur] = i * size_unfixed + k;
          edges_indexes_vec[1][compteur] = k;
          edges_weights_vec[compteur] = 1;
          edges_attr_vec[0][compteur] = 1;
          edges_attr_vec[1][compteur] = 0;
          compteur++;
        }
      }
      // print the elements of edges_indexes_vec.data()
      at::Tensor edge_first = torch::from_blob(edges_indexes_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
      at::Tensor edge_second = torch::from_blob(edges_indexes_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
      at::Tensor edges_indexes = torch::cat({edge_first, edge_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
      at::Tensor edges_attr_first = torch::from_blob(edges_attr_vec[0].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
      at::Tensor edges_attr_second = torch::from_blob(edges_attr_vec[1].data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kInt64));
      at::Tensor edges_attr = torch::cat({edges_attr_first, edges_attr_second}, 0).reshape({2, nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed});
      at::Tensor edges_weights = torch::from_blob(edges_weights_vec.data(), {nb_constraints * size_unfixed * (size_unfixed -1) / 2 + nb_constraints * size_unfixed}, torch::TensorOptions().dtype(torch::kFloat32));

      inputs.push_back(edges_indexes);
      //inputs.push_back(edges_weights);
      inputs.push_back(edges_attr.transpose(0, 1));
      at::Tensor intermediate_output = module_1.forward(inputs).toTensor();


      at::Tensor mean = intermediate_output.mean(0).repeat({nb_constraints * size_unfixed, 1});

      std::vector<torch::jit::IValue> intermediate_inputs;

      intermediate_inputs.push_back(torch::cat({intermediate_output, mean}, 1));
      at::Tensor multipliers = module_2.forward(intermediate_inputs).toTensor();

      // create a vector of multipliers
      std::vector<float> multipliers_vec;
      multipliers_vec.resize( nb_constraints * size_unfixed);
      for (int i = 0; i < nb_constraints * size_unfixed; i++) {
        multipliers_vec[i] = multipliers[i].item<float>();
      }

      float final_bound = 0.0f;
      std::vector<SubProblem> subproblems;

        // we create one subproblem for each knapsack constraint
      for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
        SubProblem subproblem;
        subproblem.weights_sub = new int[nb_items];
        subproblem.val_sub = new float[nb_items];
        subproblem.capacity = spec.capacity(idx_constraint, nb_items);

        for (int i = 0; i < fixed_variables.size(); i++) {
          subproblem.weights_sub[i] = spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints);

          if (idx_constraint == 0) {
            subproblem.val_sub[i] = spec.profit(fixed_variables[i]);
          }

          else {
            subproblem.val_sub[i] = 0.0f;
          }
        }
        for (int i = 0; i < not_fixed_variables.size(); i++) {
          subproblem.weights_sub[i + fixed_variables.size()] = spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);

          if (idx_constraint == 0) {
            float mult = 0.0f;
            for (int j = 1; j < nb_constraints; j++) {
              mult += multipliers_vec[i + size_unfixed * j];
            }
            subproblem.val_sub[i + fixed_variables.size()] = spec.profit(not_fixed_variables[i]) + mult ;
          }

          else {
            subproblem.val_sub[i + fixed_variables.size()] = - multipliers_vec[i + size_unfixed * idx_constraint];
          }
        }
        subproblem.idx_constraint = idx_constraint;
        subproblems.push_back(subproblem);
      }

      for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
        SubProblem subproblem = subproblems[id_subproblem];
        float weight_fixed = 0.0f;
    
        for (int k = 0; k < fixed_variables.size(); k++){
          weight_fixed+=subproblem.weights_sub[k]* variables[fixed_variables[k]].val();
        }

        float bound = dp_knapsack(subproblem.capacity - weight_fixed,
                                    subproblem.weights_sub + fixed_variables.size(),
                                    subproblem.val_sub + fixed_variables.size(),
                                    nb_items - fixed_variables.size(), nb_constraints,
                                    subproblem.idx_constraint,
                                    value_var_solution,
                                    not_fixed_variables,
                                    false);

        float fixed_bound = 0.0f;

        for (int k = 0; k< fixed_variables.size();k++){
            fixed_bound += subproblem.val_sub[k] * variables[fixed_variables[k]].val();
        }
        final_fixed_bounds += fixed_bound;
        final_bound += bound + fixed_bound; // sum all the bound of the knapsack sub-problem to update the multipliers
      
      }
      // We impose the constraint z <= final_bound
      rel(*this, z <= final_bound); 
   
    }
    else{
      
      for (int k=0; k<K; k++) { // We repeat the dynamic programming algo to solve the knapsack problem
                                // and at each iteration we update the value of the Lagrangian multipliers
        final_fixed_bounds = 0.0f;
        float bound_iter = 0.0f;
        std::vector<SubProblem> subproblems;

        // we create one subproblem for each knapsack constraint
        for (int idx_constraint=0; idx_constraint<nb_constraints; idx_constraint++) {
          SubProblem subproblem;
          subproblem.weights_sub = new int[nb_items];
          subproblem.val_sub = new float[nb_items];
          subproblem.capacity = spec.capacity(idx_constraint, nb_items);

          for (int i = 0; i < fixed_variables.size(); i++) {
            subproblem.weights_sub[i] = spec.weight(idx_constraint, fixed_variables[i] , nb_items, nb_constraints);

            if (idx_constraint == 0) {
              subproblem.val_sub[i] = spec.profit(fixed_variables[i]) + multipliers[fixed_variables[i]][idx_constraint];
            }

            else {
              subproblem.val_sub[i] = -multipliers[fixed_variables[i]][idx_constraint];
            }
          }
          for (int i = 0; i < not_fixed_variables.size(); i++) {
            subproblem.weights_sub[i + fixed_variables.size()] = spec.weight(idx_constraint, not_fixed_variables[i] , nb_items, nb_constraints);

            if (idx_constraint == 0) {
              subproblem.val_sub[i + fixed_variables.size()] = spec.profit(not_fixed_variables[i]) + multipliers[not_fixed_variables[i]][idx_constraint];
            }

            else {
              subproblem.val_sub[i + fixed_variables.size()] = -multipliers[not_fixed_variables[i]][idx_constraint];
            }
          }
          subproblem.idx_constraint = idx_constraint;
          subproblems.push_back(subproblem);
        }

        for (int id_subproblem=0; id_subproblem<subproblems.size(); id_subproblem++) { // iterate on all the constraints (=subproblems of the knapsack problem)
          SubProblem subproblem = subproblems[id_subproblem];
          float weight_fixed = 0.0f;
    
          for (int k = 0; k < fixed_variables.size(); k++){
            weight_fixed+=subproblem.weights_sub[k]* variables[fixed_variables[k]].val();
          }

          float bound = dp_knapsack(subproblem.capacity - weight_fixed,
                                    subproblem.weights_sub + fixed_variables.size(),
                                    subproblem.val_sub + fixed_variables.size(),
                                    nb_items - fixed_variables.size(), nb_constraints,
                                    subproblem.idx_constraint,
                                    value_var_solution,
                                    not_fixed_variables,
                                    false);

          float fixed_bound = 0.0f;

          for (int k = 0; k< fixed_variables.size();k++){
              fixed_bound += subproblem.val_sub[k] * variables[fixed_variables[k]].val();
          }
          final_fixed_bounds += fixed_bound;

          bound_iter += bound + fixed_bound; // sum all the bound of the knapsack sub-problem to update the multipliers
      
        }
        final_bound = std::min(final_bound, bound_iter);
        bound_test[k] = bound_iter;

        // Update the multipliers (Quentin method with constant learning rate) TODO : implement the article method with adaptive learning rate

        for (int i = 0; i < rows; ++i) {
          float sum = 0;
          for (int j = 1; j < cols; ++j) {
            multipliers[i][j] = multipliers[i][j] -  learning_rate *  (value_var_solution[i][0] - value_var_solution[i][j]);
            sum += multipliers[i][j];
          }
          multipliers[i][0] = sum;
        }

        //std::cout << "Iteration " << k << " : " << final_bound << std::endl;
        // We impose the constraint z <= final_bound
        rel(*this, z <= final_bound); 
      }
    }

    node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+2] = final_fixed_bounds;
    node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+2 + 1] = final_bound;
    if (write_samples) {
      if ((set_nodes.count(dicstr)==0)  ) { // and (size_unfixed>=5)  and (size_unfixed<=30)
          set_nodes.insert(dicstr);

          if (outputFileMK->is_open()) {
          for (int i=0;i<size_unfixed*(nb_constraints+1)+nb_constraints+3;i++) {
              *outputFileMK << node_problem[i]<<",";
            }

          *outputFileMK << node_problem[size_unfixed*(nb_constraints+1)+nb_constraints+3]<<"\n";
          } 
        } 
      }

    for (int i = 0; i < rows; ++i) {
      delete[] value_var_solution[i];
    }

    delete[] value_var_solution;
  }

  static void post(Space& home) {
    static_cast<MultiKnapsack&>(home).more();
  }

  /// Return cost
  virtual IntVar cost(void) const {
    return z;
  }
  /// Constructor for cloning \a s
  MultiKnapsack(MultiKnapsack& s)
    : IntMaximizeSpace(s), spec(s.spec) {
    variables.update(*this, s.variables);
    z.update(*this, s.z);
    this->order_branching = s.order_branching;
    this->activate_bound_computation = s.activate_bound_computation;
    this->activate_learning_prediction = s.activate_learning_prediction;
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
    return new MultiKnapsack(*this);
  }

  virtual void constrain(const Space& _b) { // compute the bound at each leaf node giving a solution
    const MultiKnapsack& b = static_cast<const MultiKnapsack&>(_b);
    std::cout<<"solution: "<<b.z<<std::endl;
    // We impose the constraint z >= current sol
    rel(*this, z >= b.z);
}

float dp_knapsack(int capacity,
                  int weights[],
                  float val[],
                  int nb_items,
                  int nb_constraints,
                  int idx_constraint,
                  int** value_var_solution,
                  std::vector<int>& not_fixed_variables,
                  bool verbose=false) {
    std::vector<std::vector<float>> dp(capacity + 1, std::vector<float>(nb_items + 1, 0));
    for (int i = 1; i <= nb_items; ++i) {
        for (int w = 0; w <= capacity; ++w) {
            if (weights[i - 1] <= w) {
                dp[w][i] = std::max(dp[w][i - 1], dp[w - weights[i - 1]][i - 1] + val[i - 1]);
            } else {
                dp[w][i] = dp[w][i - 1];
            }
        }
    }
    // Backtracking to find selected items
    int w = capacity;
    for (int i = nb_items; i > 0; --i) {
        if (dp[w][i] != dp[w][i - 1]) {
            value_var_solution[not_fixed_variables[i - 1]][idx_constraint] = 1;
            w -= weights[i - 1];
        }
    }
    return dp[capacity][nb_items];
}

  /// Print solution
  virtual void
  print(std::ostream& os) const {
    os << "z: " << z << std::endl;
    os << "variables: " << variables << std::endl;
  }
};

int main(int argc, char* argv[]) {

  if (argc != 2) {
     std::cerr << "usage: example-app </Users/dariusdabert/Downloads/traced_resnet_model.pt>\n";
     //return -1;
   }

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().

    module_1 = torch::jit::load(argv[1]);
    module_2 = torch::jit::load(argv[2]);

  }
  catch (const c10::Error& e) {
    std::cerr << "error with forward pass \n";
    //return -1;
  }

  std::cout << "ok\n";

  bool activate_bound_computation = true;
  bool activate_learning_prediction = true;
  bool activate_heuristic = true;
  bool write_samples = false;
  int K = 60;
  float learning_rate = 0.2f;
  float init_value_multipliers = 1.0f;

  for (int i = 0; i < 1; i++) {
    std::ifstream inputFilea("src/problem/mknapsack/solving/benchmark/" + (std::string)name[i]+ ".txt");
    if (write_samples){
      std::ofstream outputFilea((std::string)name[i]+".txt");

      std::string line;
      std::vector<int> problem;
      std::vector<int> numbers;
      int j =1;

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
        OptionsKnapsack opt=OptionsKnapsack(activate_bound_computation, activate_learning_prediction, activate_heuristic, K,learning_rate,init_value_multipliers, &outputFilea , problem, true);
        opt.instance();
        opt.solutions(0);
        opt.parse(argc,argv);
        IntMaximizeScript::run<MultiKnapsack,BAB,OptionsKnapsack>(opt);
        j++;
        }
        outputFilea.close();

    }

    else{
      std::string line;
      std::vector<int> problem;
      std::vector<int> numbers;
      int j =1;

      while (std::getline(inputFilea, line)) {
        std::vector<int> problem;
        std::istringstream iss(line);
        std::string substring;
        while (std::getline(iss, substring, ',')) {
                problem.push_back(std::stoi(substring));
            }
        std::cout<<""<<std::endl;
        OptionsKnapsack opt=OptionsKnapsack(activate_bound_computation, activate_learning_prediction, activate_heuristic, K,learning_rate,init_value_multipliers, NULL , problem, false);
        opt.instance();
        opt.solutions(0);
        opt.parse(argc,argv);
        IntMaximizeScript::run<MultiKnapsack,BAB,OptionsKnapsack>(opt);
        }
        j++;
    }
    if (write_samples)
      std::cout<<"separateur_de_probleme"<<std::endl;
    inputFilea.close(); // Close the file when done
  }
  return 0;
}

namespace {
int toy[]={2, 3,
    2, 3, 4,
    46, 76,
    12, 19, 30,
    49, 40, 31};

  int n1c1w1_a[] = {
       5, 40,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  480, 760, 800, 1185, 1200,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
 
  5605
  };

  int n1c1w1_b[] = {
      
 5, 30,
 360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
 670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
 120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
 400, 500, 500, 600, 600,
 7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
 42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
 8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
 26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
 97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
 3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
 17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
 19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
 21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
 10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
 7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
 94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
 44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
 11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 
 4554
  };

  int n1c1w1_c[] = {
    5, 30,
    360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
    670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
    120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
    370, 650, 460, 980, 870,
    7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
    59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
    8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
    26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
    97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
    3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
    17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
    19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
    21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
    10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
    7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
    94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
    44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
    11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
    94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
    30, 15, 12, 24, 90, 25, 39, 47, 98, 83
  };

  int n1c1w1_d[] = {
    5, 30,
    360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
    670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
    120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
    480, 800, 500, 300, 620,
    7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
    59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
    8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
    26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
    97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
    3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
    17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
    19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
    21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
    10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
    7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
    94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
    44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
    11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
    94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
    30, 15, 12, 24, 90, 25, 39, 47, 98, 83
  };

  int n1c1w1_e[] = {
  5, 30,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  540, 270, 500, 500, 750,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 
  4561
 
  };

  int n1c1w1_f[] = {

  5, 30,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  540, 240, 480, 600, 790,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 
  4514
 
  };

  int n1c1w1_g[] = {
  5, 40,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  480, 760, 800, 1180, 940,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 
  5557
 
  };
  int n1c1w1_h[] = {
  5, 40,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  480, 600, 700, 1200, 1200,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 
  5567
 
  };

  int n1c1w1_i[] = {
  5, 40,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  480, 760, 800, 1185, 1200,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 
  5605
 
  };

  int n1c1w1_j[] = {
  5, 40,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  750, 870, 360, 800, 940,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 
  5246
 
  };

  int n1c1w1_k[] = {
  5, 50,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  850, 1400, 1500, 450, 1100,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
 
  6339
 
  };

  int n1c1w1_l[] = {
  5, 50,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  880, 1340, 1360, 300, 1000,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
 
  5643
 
  };

  int n1c1w1_m[] = {
  5, 50,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  850, 1400, 1500, 440, 1100,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
 
  6339
 
  };

  int n1c1w1_n[] = {

  5, 50,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  850, 1400, 1500, 400, 1100,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
 
 6159
 
  };

  int n1c1w1_o[] = {
  5, 60,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  1024, 1700, 1850, 510, 1310,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
 
  6954
 
  };

  int n1c1w1_p[] = {
  5, 60,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  1130, 420, 1380, 1000, 1630,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
 
  7486
 
  };

  int n1c1w1_q[] = {
  5, 60,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  1200, 1300, 630, 1100, 1400,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
 
 7289
 
  };

  int n1c1w1_r[] = {
  5, 60,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  2090, 2200, 1190, 2460, 2320,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
 
 8633
 
  };
  int n1c1w1_s[] = {
  5, 70,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  970, 1310, 1730, 2220, 2580,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
  19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
 
  9580
 
  };
  int n1c1w1_t[] = {
  5, 70,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  1200, 1920, 2330, 620, 1460,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
  19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
 
 7698
 
  };

  int n1c1w1_u[] = {
  5, 70,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  1320, 700, 1730, 1954, 1810,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
 94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
 44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
 11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
 30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
 56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
 19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
 
 9450
 
  };
  int n1c1w1_v[] = {
  5, 70,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  1320, 600, 1730, 1954, 1810,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
  19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
 
 9074
 
  };

  int n1c1w1_w[] = {
 5, 80,
 360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
 670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
 120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
 514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
 85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
 94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
 73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
 123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
 1347, 2180, 2683, 838, 1788,
 7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
 42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
 2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
 86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
 48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
 15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
 69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
 8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
 26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
 97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
 0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
 78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
 5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
 15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
 56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
 3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
 17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
 19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
 60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
 41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
 25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
 69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
 52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
 21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
 10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
 7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
 4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
 15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
 41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
 51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
 3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
 94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
 44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
 11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
 94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
 30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
 56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
 19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
 41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
 8947
  };
  int n1c1w1_x[] = {
    5, 80,
    360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
    670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
    120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
    514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
    85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
    94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
    73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
    123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
    1360, 2200, 2700, 700, 1700,
    7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
    59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
    2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
    86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
    48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
    15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
    69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
    8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
    26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
    97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
    0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
    78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
    5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
    15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
    56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
    3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
    17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
    19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
    60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
    41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
    25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
    69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
    52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
    21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
    10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
    7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
    4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
    15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
    41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
    51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
    3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
    94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
    44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
    11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
    94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
    30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
    56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
    19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
    41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
   
    8344
   
  };
  int n1c1w1_y[] = {
  5, 80,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
  1100, 1500, 2000, 2500, 3000,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
 30, 15, 12, 24, 90 ,25 ,39 ,47, 98 ,83,
 56 ,36, 6 ,66 ,89 ,45, 38 ,1 ,18, 88,
 19, 39 ,20 ,1, 7 ,34, 68 ,32, 31, 58,
 41, 99 ,92 ,67 ,33 ,26, 25 ,68, 37, 6,
 
 10220
 
  };

int n1c1w1_z[] = {
  5, 80,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
  1500, 800, 2000, 2200, 2100,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
  19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
  41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
  9939
};


int n1c1w1_a_1[] = {
  5, 90,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
  100, 72, 31, 29, 316, 244, 70, 82, 90, 52,
  1600, 2500, 3000, 850, 2000,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
  20, 8, 4, 43, 17, 25, 36, 60, 84, 40,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
  74, 26, 10, 85, 63, 31, 22, 9, 92, 18,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
  3, 32, 82, 20, 2, 51, 18, 42, 4, 26,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
  7, 86, 48, 2, 1, 15, 74, 80, 57, 16,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
  19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
  41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
  11, 17, 48, 79, 63, 77, 17, 29, 18, 60,
 
  9584
 
  };
 int n1c1w1_b_1[] = {
  5, 90,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
  100, 72, 31, 29, 316, 244, 70, 82, 90, 52,
  1600, 2500, 3000, 900, 2000,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
  20, 8, 4, 43, 17, 25, 36, 60, 84, 40,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
  74, 26, 10, 85, 63, 31, 22, 9, 92, 18,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
  3, 32, 82, 20, 2, 51, 18, 42, 4, 26,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
  7, 86, 48, 2, 1, 15, 74, 80, 57, 16,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
  19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
  41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
  11, 17, 48, 79, 63, 77, 17, 29, 18, 60,
  9819
};


  int n1c1w1_c_1[] = {
  5, 90,
  360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
  670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
  120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
  514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
  85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
  94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
  73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
  123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
  100, 72, 31, 29, 316, 244, 70, 82, 90, 52,
  1500, 2500, 3000, 820, 2000,
  7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
  59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
  42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
  2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
  86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
  48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
  15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
  69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
  20, 8, 4, 43, 17, 25, 36, 60, 84, 40,
  8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
  26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
  97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
  0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
  78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
  5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
  15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
  56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
  74, 26, 10, 85, 63, 31, 22, 9, 92, 18,
  3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
  17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
  19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
  60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
  41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
  25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
  69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
  52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
  3, 32, 82, 20, 2, 51, 18, 42, 4, 26,
  21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
  10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
  7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
  4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
  15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
  41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
  51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
  3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
  7, 86, 48, 2, 1, 15, 74, 80, 57, 16,
  94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
  44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
  11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
  94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
  30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
  56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
  19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
  41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
  11, 17, 48, 79, 63, 77, 17, 29, 18, 60,
 
  9492
 
  };
  int n1c1w1_d_1[] = {
    5, 90,
    360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
    670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
    120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
    514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
    85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
    94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
    73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
    123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
    100, 72, 31, 29, 316, 244, 70, 82, 90, 52,
    1600, 2500, 3000, 800, 2000,
    7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
    59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
    2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
    86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
    48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
    15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
    69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
    20, 8, 4, 43, 17, 25, 36, 60, 84, 40,
    8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
    26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
    97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
    0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
    78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
    5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
    15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
    56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
    74, 26, 10, 85, 63, 31, 22, 9, 92, 18,
    3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
    17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
    19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
    60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
    41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
    25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
    69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
    52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
    3, 32, 82, 20, 2, 51, 18, 42, 4, 26,
    21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
    10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
    7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
    4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
    15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
    41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
    51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
    3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
    7, 86, 48, 2, 1, 15, 74, 80, 57, 16,
    94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
    44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
    11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
    94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
    30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
    56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
    19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
    41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
    11, 17, 48, 79, 63, 77, 17, 29, 18, 60,
   
    9410
   
  };

    int n1c1w1_e_1[] = {
    5, 90,
    360, 83, 59, 130, 431, 67, 230, 52, 93, 125,
    670, 892, 600, 38, 48, 147, 78, 256, 63, 17,
    120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
    514, 28, 87, 73, 78, 15, 26, 78, 210, 36,
    85, 189, 274, 43, 33, 10, 19, 389, 276, 312,
    94, 68, 73, 192, 41, 163, 16, 40, 195, 138,
    73, 152, 400, 26, 14, 170, 205, 57, 369, 435,
    123, 25, 94, 88, 90, 146, 55, 29, 82, 74,
    100, 72, 31, 29, 316, 244, 70, 82, 90, 52,
    2100, 1100, 3300, 3700, 3600,
    7, 0, 30, 22, 80, 94, 11, 81, 70, 64,
    59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84,
    2, 4, 18, 56, 7, 29, 93, 44, 71, 3,
    86, 66, 31, 65, 0, 79, 20, 65, 52, 13,
    48, 14, 5, 72, 14, 39, 46, 27, 11, 91,
    15, 25, 0, 94, 53, 48, 27, 99, 6, 17,
    69, 43, 0, 57, 7, 21, 78, 10, 37, 26,
    20, 8, 4, 43, 17, 25, 36, 60, 84, 40,
    8, 66, 98, 50, 0, 30, 0, 88, 15, 37,
    26, 72, 61, 57, 17, 27, 83, 3, 9, 66,
    97, 42, 2, 44, 71, 11, 25, 74, 90, 20,
    0, 38, 33, 14, 9, 23, 12, 58, 6, 14,
    78, 0, 12, 99, 84, 31, 16, 7, 33, 20,
    5, 18, 96, 63, 31, 0, 70, 4, 66, 9,
    15, 25, 2, 0, 48, 1, 40, 31, 82, 79,
    56, 34, 3, 19, 52, 36, 95, 6, 35, 34,
    74, 26, 10, 85, 63, 31, 22, 9, 92, 18,
    3, 74, 88, 50, 55, 19, 0, 6, 30, 62,
    17, 81, 25, 46, 67, 28, 36, 8, 1, 52,
    19, 37, 27, 62, 39, 84, 16, 14, 21, 5,
    60, 82, 72, 89, 16, 5, 29, 7, 80, 97,
    41, 46, 15, 92, 51, 76, 57, 90, 10, 37,
    25, 93, 5, 39, 0, 97, 6, 96, 2, 81,
    69, 4, 32, 78, 65, 83, 62, 89, 45, 53,
    52, 76, 72, 23, 89, 48, 41, 1, 27, 19,
    3, 32, 82, 20, 2, 51, 18, 42, 4, 26,
    21, 40, 0, 6, 82, 91, 43, 30, 62, 91,
    10, 41, 12, 4, 80, 77, 98, 50, 78, 35,
    7, 1, 96, 67, 85, 4, 23, 38, 2, 57,
    4, 53, 0, 33, 2, 25, 14, 97, 87, 42,
    15, 65, 19, 83, 67, 70, 80, 39, 9, 5,
    41, 31, 36, 15, 30, 87, 28, 13, 40, 0,
    51, 79, 75, 43, 91, 60, 24, 18, 85, 83,
    3, 85, 2, 5, 51, 63, 52, 85, 17, 62,
    7, 86, 48, 2, 1, 15, 74, 80, 57, 16,
    94, 86, 80, 92, 31, 17, 65, 51, 46, 66,
    44, 3, 26, 0, 39, 20, 11, 6, 55, 70,
    11, 75, 82, 35, 47, 99, 5, 14, 23, 38,
    94, 66, 64, 27, 77, 50, 28, 25, 61, 10,
    30, 15, 12, 24, 90, 25, 39, 47, 98, 83,
    56, 36, 6, 66, 89, 45, 38, 1, 18, 88,
    19, 39, 20, 1, 7, 34, 68, 32, 31, 58,
    41, 99, 92, 67, 33, 26, 25, 68, 37, 6,
    11, 17, 48, 79, 63, 77, 17, 29, 18, 60,
 
    11191
  };

  int* mknps[] = {
    &toy[0],
    &n1c1w1_b[0],
    &n1c1w1_c[0],
    &n1c1w1_d[0],
    &n1c1w1_e[0],
    &n1c1w1_f[0],
    &n1c1w1_g[0],
    &n1c1w1_h[0],
    &n1c1w1_i[0],
    &n1c1w1_j[0],
    &n1c1w1_k[0],
    &n1c1w1_l[0],
    &n1c1w1_m[0],
    &n1c1w1_n[0],
    &n1c1w1_o[0],
    &n1c1w1_p[0],
    &n1c1w1_q[0],
    &n1c1w1_r[0],
    &n1c1w1_s[0],
    &n1c1w1_t[0],
    &n1c1w1_u[0],
    &n1c1w1_v[0],
    &n1c1w1_w[0],
    &n1c1w1_x[0],
    &n1c1w1_y[0],
    &n1c1w1_z[0],
    &n1c1w1_a_1[0],
    &n1c1w1_b_1[0],
    &n1c1w1_c_1[0],
    &n1c1w1_d_1[0],
    &n1c1w1_e_1[0]
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

