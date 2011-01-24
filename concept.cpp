// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * Authors: Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

/*
  This files provides all the following classes:
  - OMPObject
  - OMPPrintable
  - OMPVector
   - OMPMatrix
   - OMPMatrix3D
  - OMPOperation
   - OMPComputeUnary
    - OMPPrefix
     - OMPPrefixSum
     - OMPPrefixProduct
   - OMPComputeConstUnary
    - OMPBroadcastMatrix
   - OMPComputeBinary
   - OMPComputeConstBinary
    - OMPComputeBinaryAtom
     - OMPAddition
     - OMPMultiply
  - OMPRandom
  - OMPRngVector
  - OMPRngMatrix
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <functional>
#include <vector>
#include <omp.h>

/* some macros needed for prefix sum algorithm */
#define EVEN(x)	( ( (x) % 2 ) == 0 )
#define ODD(x)	( ( (x) % 2 ) == 1 )

/* this is the common base class for all classes */
class OMPObject {};

/* all classes implementing OMPPrintable must define printOn
   function. This allows to print some information about
   the class's data.
 */
class OMPPrintable
{
public:
    virtual ~OMPPrintable() {}

    virtual void printOn( std::ostream& ) const = 0;
};

std::ostream& operator<<( std::ostream& os, const OMPPrintable& o )
{
    o.printOn( os );
    return os;
}

/* here's the vector data structure inheriting from std::vector
   we have removed all features like iterator which does not work
   with OpenMP 2.5. This is going to be available with the next
   stardart version 3.0.
   At this time we can use subscript in using the size method
   and iterating from 0 to size - 1.
 */
template < typename Atom >
class OMPVector : public OMPObject, public OMPPrintable, private std::vector< Atom >
{
public:
    using std::vector< Atom >::clear;
    using std::vector< Atom >::resize;
    using std::vector< Atom >::size;
    using std::vector< Atom >::operator[];
    using std::vector< Atom >::push_back;

    OMPVector( size_t size = 0, const Atom& value = Atom() ) : std::vector< Atom >()
    {
	if ( size == 0 ) return;

	resize( size );

#pragma omp parallel for shared( size )
	for ( size_t i = 0; i < size; ++i )
	    {
		operator[]( i ) = value;
	    }
    }

    OMPVector( const OMPVector< Atom >& v ) : std::vector< Atom >() { *this = v; }

    OMPVector& operator=( const OMPVector< Atom >& v )
    {
	size_t size = v.size();

	if ( size == 0 ) return *this;

	resize( size );

#pragma omp parallel for shared( size )
	for ( size_t i = 0; i < size; ++i )
	    {
		operator[]( i ) = v[ i ];
	    }

	return *this;
    }

    void printOn( std::ostream& os ) const
    {
	size_t size = this->size();

	if ( size == 0 ) return;

	os << "[" << operator[]( 0 );
	for ( size_t i = 1; i < size; ++i )
	    {
		os << ", " << operator[]( i );
	    }
	os << "]";
    }
};

/* here's the matrix data structure */
template < typename Atom >
class OMPMatrix : public OMPVector< OMPVector< Atom > >
{
public:
    using OMPVector< OMPVector< Atom > >::clear;
    using OMPVector< OMPVector< Atom > >::resize;
    using OMPVector< OMPVector< Atom > >::size;
    using OMPVector< OMPVector< Atom > >::operator[];
    using OMPVector< OMPVector< Atom > >::push_back;
    using OMPVector< OMPVector< Atom > >::operator=;

    OMPMatrix() : OMPVector< OMPVector< Atom > >(), _nrows( 0 ), _ncols( 0 ) {}

    OMPMatrix( size_t size, const Atom& value = Atom() ) : OMPVector< OMPVector< Atom > >( size, OMPVector< Atom >( size, value ) ), _nrows( size ), _ncols( size ) {}

    OMPMatrix( size_t nrows, size_t ncols, const Atom& value = Atom() ) : OMPVector< OMPVector< Atom > >( nrows, OMPVector< Atom >( ncols, value ) ), _nrows( nrows ), _ncols( ncols ) {}

    OMPMatrix( const OMPMatrix< Atom >& m ) : OMPVector< OMPVector< Atom > >( m ) {}

    // friend OMPMatrix< Atom > operator*( const OMPMatrix< Atom >&, const OMPMatrix< Atom >& ) const;

    size_t nrows() const { return _nrows; }
    size_t ncols() const { return _ncols; }

    void resize( size_t nrows, size_t ncols = nrows )
    {
	_nrows = nrows;
	_ncols = ncols;

	resize( _nrows );

#pragma omp parallel for
	for ( size_t i = 0; i < _nrows; ++i )
	    {
		operator[]( i ).resize( _ncols );
	    }
    }

    void printOn( std::ostream& os ) const
    {
	size_t size = this->size();

	if ( size == 0 ) return;

	os << "[" << operator[]( 0 );
	for ( size_t i = 1; i < size; ++i )
	    {
		os << std::endl << " " << operator[]( i );
	    }
	os << "]";
    }

private:
    size_t _nrows;
    size_t _ncols;
};

/* here's the 3 dimentional matrix data structure */
template < typename Atom >
class OMPMatrix3D : public OMPVector< OMPMatrix< Atom > >
{
public:
    using OMPVector< OMPMatrix< Atom > >::clear;
    using OMPVector< OMPMatrix< Atom > >::resize;
    using OMPVector< OMPMatrix< Atom > >::size;
    using OMPVector< OMPMatrix< Atom > >::operator[];
    using OMPVector< OMPMatrix< Atom > >::push_back;
    using OMPVector< OMPMatrix< Atom > >::operator=;

    OMPMatrix3D( size_t size = 0, const OMPMatrix< Atom >& value = OMPMatrix< Atom >() ) : OMPVector< OMPMatrix< Atom > >( size, value ) {}

    OMPMatrix3D( const OMPMatrix3D< Atom >& m3d ) : OMPVector< OMPMatrix< Atom > >( m3d ) {}

    void printOn( std::ostream& os ) const
    {
	size_t size = this->size();

	if ( size == 0 ) return;

	os << "[3D matrix not printable yet]";
    }
};

/* here's some typedefs to make vectors and matrices parameters passing more understandable thanks to Numerical Recipes book's tips.
 */
typedef OMPVector< double > OMPVecDoubl;
typedef OMPVector< int > OMPVecInt;

typedef const OMPVecDoubl OMPVecDoubl_I;
typedef OMPVecDoubl OMPVecDoubl_O, OMPVecDoubl_IO;

typedef const OMPVecInt OMPVecInt_I;
typedef OMPVecInt OMPVecInt_O, OMPVecInt_IO;

typedef OMPMatrix< double > OMPMatrixDoubl;
typedef OMPMatrix< int > OMPMatrixInt;

typedef const OMPMatrixDoubl OMPMatrixDoubl_I;
typedef OMPMatrixDoubl OMPMatrixDoubl_O, OMPMatrixDoubl_IO;

typedef const OMPMatrixInt OMPMatrixInt_I;
typedef OMPMatrixInt OMPMatrixInt_O, OMPMatrixInt_IO;

/* this is our base class for all operator classes */
class OMPOperation : public OMPObject {};

/* here's an overload of the STL class unary_function for our unary computing operators
   there is a virtual pure method that all derivated classes must define:
   virtual Result operator()( Arg1 ) = 0;
 */
template < typename Arg1, typename Result >
class OMPComputeUnary : public OMPOperation, public std::unary_function< Arg1, Result > {};

/* the const version */
template < typename Arg1, typename Result >
class OMPComputeConstUnary : public OMPOperation
{
public:
    virtual ~OMPComputeConstUnary() {}
    virtual Result operator()( Arg1 ) const = 0;
};

/* here's an overload of the STL class binary_function for our binary computing operators
   there is a virtual pure method that all derivated classes must define:
   virtual Result operator()( Arg1, Arg2 ) = 0;
 */
template < typename Arg1, typename Arg2, typename Result >
class OMPComputeBinary : public OMPOperation, public std::binary_function< Arg1, Arg2, Result > {};

/* the const version */
template < typename Arg1, typename Arg2, typename Result >
class OMPComputeConstBinary : public OMPOperation
{
public:
    virtual ~OMPComputeConstBinary() {}
    virtual Result operator()( Arg1, Arg2 ) const = 0;
};

/* here's the abstract class of an operator with two Atom parameters */
template < typename Atom >
class OMPComputeBinaryAtom : public OMPComputeConstBinary< const Atom&, const Atom&, Atom > {};

/* here's the addition */
template < typename Atom >
class OMPAddition : public OMPComputeBinaryAtom< Atom >
{
public:
    Atom operator()( const Atom& a, const Atom& b) const { return a + b; }
};

/* here's the multiplication */
template < typename Atom >
class OMPMultiply : public OMPComputeBinaryAtom< Atom >
{
public:
    Atom operator()( const Atom& a, const Atom& b) const { return a * b; }
};

/* And some others like ... Sub, Divide, ... */

/* here's an operator to broadcast in using the EREW model */
template < typename Atom >
class OMPBroadcastMatrix : public OMPComputeConstUnary< const OMPMatrix< Atom >&, OMPMatrix3D< Atom > >
{
public:
    OMPMatrix3D< Atom > operator()( const OMPMatrix< Atom >& m ) const
    {
	OMPMatrix3D< Atom > M3D( m.nrows(), m );
	return M3D;

	size_t nrows = m.nrows();
	size_t ncols = m.ncols();

	OMPMatrix3D< Atom > m3d( nrows );

	if ( nrows == 0 ) return m3d;
	if ( ncols == 0 ) return m3d;

	m3d[0] = m;

	size_t size = ::log2( nrows );

#pragma omp parallel for shared( size )
	for ( size_t s = 0; s < size; ++s )
	    {
		for ( size_t h = 0, end = s*s; h < end; ++h )
		    {
			m3d[(1 << s) + h] = m3d[h];
		    }
	    }

	return m3d;
    }
};

/* here's a function to overload mupliciation operator and call a matrix product */
template < typename Atom >
OMPMatrix< Atom > operator*( const OMPMatrix< Atom >& A, const OMPMatrix< Atom >& B )
{
    size_t nrows = A.nrows();
    size_t ncols = A.ncols();

    OMPMatrix< Atom > C( nrows, ncols, 0 );

    if ( nrows == 0 ) return C;
    if ( ncols == 0 ) return C;

    OMPMatrix3D< Atom > A3d = OMPBroadcastMatrix< Atom >()( A );
    OMPMatrix3D< Atom > B3d = OMPBroadcastMatrix< Atom >()( B );

#pragma omp parallel shared( nrows, ncols, A3d, B3d, C )
    {
#pragma omp parallel for
	for ( size_t i = 0; i < nrows; ++i )
	    {
		for ( size_t j = 0; j < ncols; ++j )
		    {
			for ( size_t k = 0; k < nrows; ++k )
			    {
				C[i][j] += A3d[k][i][k] * B3d[k][k][j];
			    }
		    }
	    }
    }

    return C;
}

/* here's the prefix algorithm */
template < typename Atom >
class OMPPrefix : public OMPComputeUnary< const OMPVector< Atom >&, Atom >
{
public:
    OMPPrefix( const OMPComputeBinaryAtom< Atom >& op ) : _op(op) {}

    Atom operator()( const OMPVector< Atom >& vec )
    {
	size_t size = vec.size();

	if ( size == 0 ) return 0;

	OMPMatrix< Atom > B( size );
	OMPMatrix< Atom > C( size );

	B[0] = vec;

	for ( size_t h = 1; h <= ::log2(size); ++h )
	    {
		unsigned int subsize = size / (1 << h);

#pragma omp parallel for shared(subsize)
		for ( size_t k = 1; k <= subsize; k++ )
		    {
			B[h][k] = _op( B[h - 1][2 * k - 1], B[h - 1][2 * k] );
		    }
	    }

	for ( int h = ::log2(size); h >= 0; --h )
	    {
		size_t subsize = size / (1 << h);

#pragma omp parallel for shared(subsize)
		for ( size_t k = 0; k < subsize; ++k )
		    {
			if ( EVEN(k) ) { C[h][k] = C[h + 1][k / 2]; }
			if ( k == 1 ) { C[h][1] = B[h][1]; }
			if ( ODD(k) ) { C[h][k] = _op( C[h + 1][k - 1], B[h][k] ); }
		    }
	    }

	return B[::log2(size)][1];
    }

private:
    const OMPComputeBinaryAtom< Atom >& _op;
};

/* this is the prefix sum inheriting from prefix algorithm */
template < typename Atom >
class OMPPrefixSum : public OMPPrefix< Atom >
{
public:
    OMPPrefixSum() : OMPPrefix< Atom >( _add ) {}

private:
    OMPAddition< Atom > _add;
};

/* this is the prefix product inheriting from prefix algorithm */
template < typename Atom >
class OMPPrefixProduct : public OMPPrefix< Atom >
{
public:
    OMPPrefixProduct() : OMPPrefix< Atom >( _mul ) {}

private:
    OMPMultiply< Atom > _mul;
};

/* And some others like ... PrefixMul, PrefixDivide, ... */

/* a wrapper for random number generator function */
template < typename Atom,
	   typename OMPRandomFunc = int (*)(),
	   typename OMPSeedRandomFunc = void (*)( unsigned int ) >
class OMPRandom : public OMPObject, public OMPPrintable
{
public:
    OMPRandom( const OMPRandomFunc rand, const OMPSeedRandomFunc srand, Atom max = 1, unsigned int seed = ::time(NULL) )
	: _rand( rand ), _srand( srand ), _max( max )
    {
	(*_srand)( seed );
    }

    void reseed( unsigned int seed ) const { (*_srand)( seed ); }

    Atom operator()() const { return _max * ( (double)(*_rand)() / RAND_MAX ); }

    void printOn( std::ostream& os ) const
    {
	os << "{"
	   // << "@rand: " << (void*)_rand
	   // << ", @srand: " << (void*)_srand
	   << ", max: " << _max
	   << "}";
    }

private:
    const OMPRandomFunc& _rand;
    const OMPSeedRandomFunc& _srand;
    Atom _max;
};

/* here's a class using a random number generator to apply a random number to all the elements of a vector */
template < typename Atom >
class OMPRngVector : public OMPObject
{
public:
    OMPRngVector( OMPRandom< Atom >& rand ) : _rand( rand ) {};

    void operator()( OMPVector< Atom >& vec, size_t begin = 0, size_t end = 0 )
    {
	size_t size = vec.size();

	if ( size == 0 ) return;
	if ( end == 0 ) end = size;

#pragma omp parallel shared(begin, end)
	{
	    size_t rank = omp_get_num_threads();

#pragma omp for
	    for ( size_t i = begin; i < end; ++i )
		{
		    _rand.reseed( i + rank );
		    vec[i] = _rand();
		}
	}
    }

private:
    const OMPRandom< Atom >& _rand;
};

/* here's a class using a random number generator to apply a random number to all the elements of a matrix */
template < typename Atom >
class OMPRngMatrix : public OMPObject
{
public:
    OMPRngMatrix( OMPRandom< Atom >& rand ) : _rand( rand ) {};

    void operator()( OMPMatrix< Atom >& m, size_t lbegin = 0, size_t lend = 0, size_t cbegin = 0, size_t cend = 0 )
    {
	size_t nrows = m.nrows();
	size_t ncols = m.ncols();

	if ( nrows == 0 ) return;
	if ( ncols == 0 ) return;
	if ( lend == 0 ) lend = nrows;
	if ( cend == 0 ) cend = ncols;

#pragma omp parallel shared(lbegin, lend, cbegin, cend)
	{
	    size_t rank = omp_get_num_threads();

#pragma omp for
	    for ( size_t i = lbegin; i < lend; ++i )
		{
		    for ( size_t j = cbegin; j < cend; ++j )
			{
			    _rand.reseed( i * lend + j + rank );
			    m[i][j] = _rand();
			}
		}
	}
    }

private:
    const OMPRandom< Atom >& _rand;
};

int main(void)
{
    OMPRandom< int > rng( ::rand, ::srand, 10, 12 );
    std::cout << "here's rng data structure " << rng << std::endl;

    OMPRngVector< int > rngvec( rng );
    OMPVecInt vec( 8 );

    std::cout << "here's vec " << vec << std::endl;
    rngvec( vec, 1 );
    std::cout << "here's vec " << vec << std::endl;

    OMPPrefixSum< int > ps;
    std::cout << "result of prefix sum: " << ps( vec ) << std::endl;

    OMPMatrixInt A(8);
    OMPMatrixInt B(8, 1);
    OMPRngMatrix< int > rngmatrix( rng );

    rng.reseed( 42 );
    rngmatrix( A );

    OMPMatrix< int > C = A * B;
    std::cout << C << std::endl;

    OMPMatrix3D< int > matrix3D( 10, C );
    std::cout << matrix3D << std::endl;

    return 0;
 }
