#pragma once
#include <vector>
#include <cassert>

/**
 * @struct 
 * @brief Iterator ensemble to traverse Tensor data and gradient; 
 * 
 * View contains two pairs of Iterators spanning contignuous values in a `std::vector<float>`.
 * Per intend the Iterator pairs should represent data and its gradient respectively.
 * View is designed to allow functional layers access to data from both Tensor and other classes constaining `std::vector<float>`.
 * @note To ensure well defined behaviour, iterators should not exceed the domain of viewed `std::vector<float>`.
 */
struct View
{
   friend class Batch;
   using iter = std::vector<float>::iterator;

   /**
    * @brief Copy constructor.
    */
   View(const View& v) = default;

   /**
    * @brief Explicit initialization of iterators in View.
    */
   View(iter data_begin_, iter data_end_, iter grad_begin_, iter grad_end_)
   :data_begin (data_begin_),
    data_end   (data_end_  ), 
    grad_begin (grad_begin_), 
    grad_end   (grad_end_  )
   {
      assert(data_end - data_begin == grad_end - grad_begin); // assert, that Iterator pairs must have the same distance. 
      assert(data_end - data_begin >= 0); // assert, that traversal will terminate as intended.
      assert(grad_end - grad_begin >= 0); // assert, that traversal will terminate as intended.  
   };

   /**
    * @brief Number of data elements.
    * @returns `size_t`.
    */
   [[nodiscard("costly")]] constexpr size_t size() const {return static_cast<size_t>(data_end - data_begin);}
   
   iter data_begin; /**< First element of data */
   iter data_end; /**< Element after last Element of data */
   iter grad_begin; /**< First element of gradient */
   iter grad_end; /**< Element after last Element of gradient */
   size_t batchsize;

private:
   constexpr void set_img(View::iter begin, View::iter end){
      data_begin = begin;
      data_end = end;
   }
};
