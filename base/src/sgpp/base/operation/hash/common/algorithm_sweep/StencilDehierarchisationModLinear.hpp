// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef STENCILDEHIERARCHISATIONMODLINEAR_HPP
#define STENCILDEHIERARCHISATIONMODLINEAR_HPP

#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
#include <sgpp/base/operation/hash/OperationStencilHierarchisation.hpp>

#include <sgpp/globaldef.hpp>


namespace sgpp {
namespace base {



/**
 * Class that implements the dehierarchisation on a linear sparse grid. Therefore
 * the ()operator has to be implement in order to use the sweep algorithm for
 * the grid traversal
 */
class StencilDehierarchisationModLinear {
 protected:
  typedef GridStorage::grid_iterator grid_iterator;

  /// the grid object
  GridStorage& storage;

 public:
  /**
   * Constructor, must be bind to a grid
   *
   * @param storage the grid storage object of the the grid, on which the dehierarchisation should be executed
   * @param surplusStencil stencil for surplus
   * @param neighborStencil stencil for neighbors
   * @param weightStencil weighted stencil
   */
  StencilDehierarchisationModLinear(
    GridStorage& storage,
    OperationStencilHierarchisation::IndexStencil& surplusStencil,
    OperationStencilHierarchisation::IndexStencil& neighborStencil,
    OperationStencilHierarchisation::WeightStencil& weightStencil);
  /**
   * Destructor
   */
  ~StencilDehierarchisationModLinear();

  /**
   * Implements operator() needed by the sweep class during the grid traversal. This function
   * is applied to the whole grid.
   *
   * @param source this DataVector holds the linear base coefficients of the sparse grid's ansatz-functions
   * @param result this DataVector holds the node base coefficients of the function that should be applied to the sparse grid
   * @param index a iterator object of the grid
   * @param dim current fixed dimension of the 'execution direction'
   */
  void operator()(DataVector& source, DataVector& result, grid_iterator& index,
                  size_t dim);


 protected:
  /**
   * Recursive dehierarchisaton algorithm, this algorithms works in-place -> source should be equal to result
   *
   * @param source this DataVector holds the linear base coefficients of the sparse grid's ansatz-functions
   * @param result this DataVector holds the node base coefficients of the function that should be applied to the sparse grid
   * @param index a iterator object of the grid
   * @param dim current fixed dimension of the 'execution direction'
   * @param seql left value of the current region regarded in this step of the recursion
   * @param seqr right value of the current region regarded in this step of the recursion
   */
  void rec(DataVector& source, DataVector& result, grid_iterator& index,
           size_t dim, int seql, int seqr);


  OperationStencilHierarchisation::IndexStencil&  _surplusStencil;
  OperationStencilHierarchisation::IndexStencil&  _neighborStencil;
  OperationStencilHierarchisation::WeightStencil& _weightStencil;
};

}  // namespace base
}  // namespace sgpp

#endif /* STENCILDEHIERARCHISATIONMODLINEAR_HPP */
