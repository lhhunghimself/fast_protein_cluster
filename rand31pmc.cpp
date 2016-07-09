#include <iostream>
#include <string>
#include <cmath>
using namespace std;

// Compile with the GCC C++ compiler:  
//
//    g++ rand31-park-miller-carta.cc  -o rand31-pmc


// rand31-park-miller-carta.cc 
// ---------------------------       
//
// Robin Whittle  rw@firstpr.com.au    2005 September 21
//
// For more information and updates: http://www.firstpr.com.au/dsp/rand31/
//
// This code contains two separate implementations of the 1988 Park Miller
// "minimal standard"  31 bit PRNG (Pseudo-Random Number Generator):  
// 
// rand31pm   The conventional Park Miller "Integer Version 1" algorithm
//            but implemented rather slowly in double-precision floating 
//            point to ensure accuracy and portability.
//            
//            This is intended as a reference.
//
// rand31pmc  My own fast integer implementation of David G. Carta's
//            approach to this PRNG, using only 32 bit integer maths
//            and no divisions.
//
// My aim is to show that the two are functionally identical, to explain 
// Carta's implementation, and to provide code which can be used by others.  
//
// Please grab the rand31pmc class and use it in your own projects.  A 
// separate pair of .c and .h files are available for incorporation into 
// ANSI C projects.
//
// When running the C versions of the Carta implementation and the 
// plain floating point Park Miller aglorithm, on an 800MHz Pentium III with
// no compiler optimisation, the Carta version produced 13 million results a 
// second (61 CPU clock-cycles per result) and the floating point version 
// was about 1/4 this speed.
// 
// The two classes  below implement the same multiplicative linear 
// congruential PRNG (pseudo-random number generator).  This type of
// PRNG is not suitable for crytpography and does not produce sequences 
// long enough or random enough for most simulation purposes.  
//
// Most of the detailed analytical work on PRNGs has gone into those 
// suitable for crypto and simulation.  PRNGs well respected in these 
// fields - such as the Mersenne Twister for simulation - are slow, tricky, 
// or require excessive memory compared with a linear congruentual 
// generator.
//
// This leaves people who are searching for a fast, well studied, linear 
// congruential PRNG looking at a varied and often troublesome bunch of 
// stuff, including algorithms written up in textbooks or incorporated in 
// libraries, many of which produce poor results.  Since it can be hard to 
// recognise the problems in a PRNG, it is best to use one which is well 
// studied and respected.
//
// Unfortunately, linear congruential generators have many bad algorithms 
// and/or bad implementations.  Most publicly available implementations 
// of the good algorithms involve division.  (The major exception I know 
// of is Ray Gardner's rg_rand.c, which implements the Carta algorithm.)
//
//
// There is a widely used 32 bit integer implementation of Park-Miller's
// "minimal standard" PRNG, using Shrage's approach: Park Miller's 
// "Integer Version 2".  This requires a division.  I don't have this 
// implementation here.  The "Integer Version 1" implementation exists 
// to provide the reference sequence of pseudo-random numbers by which we 
// test the second implementation.
//
//
// The second class "rand31pmc" is functionally identical to, and was 
// inspired by, a snippet of C code by Ray Gardner which dates from as
// early as 1995 (according to Google searches in 2005): 
//
//   http://c.snippets.org/snip_lister.php?fname=rg_rand.c
//
// This cites Carta, and implements Carta's approach, but has no explanation
// of how this approach works.  Carta's paper contains a far more complex
// explanation than the one I give below.
//
// I don't have contact details for Ray Gardner, or any of the people 
// mentioned in the references.  Eric Raymond mentions he wrote some 
// hypertext-like code with Ray Gardner in the early 90s, and I think
// Ray Gardner may have moderated Usenet comp.lang.c.
//
// There are two important aims which are achieved both by David Carta's 
// algorithm, Ray Gardner's code and class "rand31pmc":
//
// 1 - The code can run with 32 bit math operations - including a 16 x 16
//     = 32 bit multiply.
//
// 2 - There is no need for a division.
//
// David Carta's 1990 paper gave no code examples. 
//
// Stephen K. Park and Keith W. Miller rejected Carta's algorithm in 
// 1993 "... the generator as implemented by Carta is ~not~ the minimal 
// standard; it isn't even a full-period generator.  We know of no good 
// reason to use Carta's generator."
//
// But here, by compiling and running this program, you can see that 
// David Carta's algorithm produces identical results to the Park Miller 
// algorithm.
// 
// Assuming Ray Gardner really did write his code (rather than copy
// it from somewhere else), my best guess is that he started off
// trying to implement Shrage's approach (as the comments and the
// unused Shrage-specific constants suggest) and then made his code
// do what David Carta suggested.  I guess that since it worked, 
// he left it at that, without updating the comments and constants.
//
// My explanation of why Carta's algorithm works is in the code, where it 
// belongs.  Please don't strip the comments out.  
//
// It is reasonable to expect that Carta's approach would run faster 
// than Shrage's approach in "Integer Version 2" - if only because
// there is no division.  
//
// My main interest here is proving that Carta's approach works 
// properly, because I want to implement it on CPUs such as the dsPIC 
// which have a slow divide.  Most CPUs have a slow divide anyway, as 
// far as I know, so if you are in a hurry, Carta's approach is 
// probably the best.
//
// References:
//
//    Stephen K. Park and Keith W. Miller 
//    Random Number Generators: Good Ones are Hard to Find
//    Communications of the ACM, Oct 1988, Vol 31 Number 10 1192-1201
//
//       Like the other two papers, this one is normally only available
//       from the ACM site via subscription.  You should be able to
//       access this paper electronically or in print at a university
//       library.  You may also be able to find the .PDF wild on the
//       Net.  Search for "p1192-park.pdf".  For instance:
//
//         http://www-scf.usc.edu/~csci105/links/p1192-park.pdf     
//
//    David F. Carta
//    Two Fast Implementations of the "Minimal Standard" Random Number Generator
//    Communications of the ACM, Jan 1990, Vol 33 Number 1 87-88  (p87-carta.pdf)
//
//    George Marsaglia; Stephen J. Sullivan; Stephen K. Park, Keith W. Miller, 
//    Paul K. Stockmeyer
//    Remarks on Choosing and Implementing Random Number Generators 
//    Communications of the ACM, Jul 1993, Vol 36 Number 7 105-110 (p105-crawford.pdf)
//
// The following code is public domain.  If you use this code, I request that 
// you keep the comments with it, to save some poor sod from having to figure 
// out the history behind it.  


//////////////////////////////////////////////////////////////////////////////
//
// rand31pm
//
// 31 bit Pseudo Random Number Generator based on Park Miller "Integer 
// Version 1" - but done with double-precision floating point so we are not 
// concerned with the limits of integer operations.  This is not intended for 
// fast operation - but *maybe* it would be faster than the integer 
// implementation on some CPUs.  
//  
// Methods:
//    
//    seedi    Set seed with a 31 bit unsigned integer between 1 and 
//             0x7FFFFFFE inclusive.  Don't use 0!
//
//    ranlui   Provides the next pseudorandom number as a long unsigned 
//             integer (31 bits).
//
//    ranf     Provides the next pseudorandom number as a float between
//             nearly 0 and nearly 1.0.

class rand31pm {
                                    // The sole item of state - a 32 bit 
                                    // integer.
    long unsigned int seed31;   

public:
                                    // Constructor sets seed31 to 1, 
                                    // in case no seedi operation is
                                    // used.
    rand31pm() {seed31 = 1;}                                    
                                    
                                    // Declare methods.
                                    
    void              seedi(long unsigned int);
    long unsigned int ranlui(void);  
    float             ranf(void);
    

private:
                                    // nextrand()
                                    //
                                    // Generate next pseudo-random number.
                                    
                                    // Multiplier constant = 16807 = 7^5.  
                                    // Park and Miller in 1993 recommend
                                    // 48271.
    #define consta 16807            
//  #define consta 48271            
                                    // Modulus constant = 2^31 - 1 =
                                    // 0x7FFFFFFFF.  Use .0 to deter compiler
                                    // from complaining about a very large 
                                    // integer constant.    
    #define constm 2147483647.0     
                                            
    long unsigned int nextrand()
    {
        double const a = consta;
        double const m = constm;
                                    // This is the linear congrentual 
                                    // generator:
                                    //  
                                    // Multiply the old seed by constant a 
                                    // and take the modulus of the result 
                                    // (the remainder of a division) by 
                                    // constant m.
                                    
        seed31 = (long)(fmod((seed31 * a), m));

        return (seed31);        
    }
};

                                    /////////////////////////////////////////
                                    //
                                    // Implementations of the methods.
                                    
                                    // seedi()
                                    //
                                    // Set the seed from a long unsigned 
                                    // integer.  If zero is used, then
                                    // the seed will be set to 1.
                                                                        
void rand31pm::seedi(long unsigned int seedin)
{
    if (seedin == 0) seedin = 1;
    seed31 = seedin;
}
                                    
                                    // ranlui()
                                    //
                                    // Return next pseudo-random value as
                                    // a long unsigned integer.
                                    
long unsigned int rand31pm::ranlui(void)  
{
    return nextrand();
}

                                    // ranf()
                                    //
                                    // Return next pseudo-random value as
                                    // a floating point value.
float rand31pm::ranf(void)  
{
    return (nextrand() / constm);
}


//////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////
//
//
// rand31pmc
//
// Robin Whittle  2005 September 20
//
// 31 bit pseudo-random number generator based on:
//
//   Lehmer (1951)
//   Lewis, Goodman & Miller (1969)
//   Park & Miller (1983)
//   
// implemented according to the optimisation suggested by David G. Carta
// in 1990 which uses 32 bit math and does not require division.  
// Park and Miller rejected Carta's approach in 1993.  Carta provided no 
// code examples.  Carta's approach produces identical results to Park 
// and Miller's code.
//
// Copyright public domain . . . *but*:
//
// * Please leave the comments intact so inquiring minds have a chance of 
// * understanding how this implementation works and chasing the 
// * references to see the strengths and limitations of this particular 
// * pseudo-random number generator.
//
// Output is a 31 bit unsigned integer.  The range of values output is
// 1 to 2,147,483,646 and the seed must be in this range too.  The
// output sequence repeats in a loop of this length = (2^31 - 2).
//
// The output stream has some predictable patterns.  For instance, after 
// a very low output, the next one or two outputs will be relatively low 
// (compared to the 2 billion range) because the multiplier is only 16,807.
// Linear congruential generators are not suitable for cryptography or 
// simulation work (such as Monte Carlo Method), but they are probably 
// fine for many uses where the output is sound or vision for human 
// perception.  
//
// The particular generator implemented here:
//
//   New-value = (old-value * 16807) mod 0x7FFFFFFF 
//
// is probably the best studied linear congruentual PRNG.  It is not the very
// best, but it is far from the worst.
//
// For the background on this implementation, and the Park Miller
// "Minimal Standard" linear congruential PRNG, please see:
//
//    http://www.firstpr.com.au/dsp/rand31/  
//
//    Stephen K. Park and Keith W. Miller 
//    Random Number Generators: Good Ones are Hard to Find
//    Communications of the ACM, Oct 1988, Vol 31 Number 10 1192-1201
//
//    David G. Carta
//    Two Fast Implementations of the "Minimal Standard" Random Number Generator
//    Communications of the ACM, Jan 1990, Vol 33 Number 1 87-88
//
//    George Marsaglia; Stephen J. Sullivan; Stephen K. Park, Keith W. Miller, 
//    Paul K. Stockmeyer
//    Remarks on Choosing and Implementing Random Number Generators 
//    Communications of the ACM, Jul 1993, Vol 36 Number 7 105-110
//
//    http://random.mat.sbg.ac.at has lots of material on PRNG quality. 
//
//
// The sequence of values this PRNG should produce includes:
// 
//      Result     Number of results after seed of 1
//
//       16807          1
//   282475249          2
//  1622650073          3
//   984943658          4
//  1144108930          5
//   470211272          6
//   101027544          7
//  1457850878          8
//  1458777923          9
//  2007237709         10
//
//   925166085       9998
//  1484786315       9999
//  1043618065      10000
//  1589873406      10001
//  2010798668      10002
//
//  1227283347    1000000
//  1808217256    2000000
//  1140279430    3000000
//   851767375    4000000
//  1885818104    5000000
//
//   168075678   99000000
//  1209575029  100000000
//   941596188  101000000
//
//  1207672015 2147483643
//  1475608308 2147483644
//  1407677000 2147483645
//           1 2147483646
//       16807 2147483647
//
// Carta refers to two registers p (15 bits) and q (31 bits) which
// together hold the 46 bit multiplication product:
//
//         |                   |                   |                   |
//          4444 4444 3333 3333 3322 2222 2222 1111 1111 11            
//          7654 3210 9876 5432 1098 7654 3210 9876 5432 1098 7654 3210
//
//   q 31                        qqq qqqq qqqq qqqq qqqq qqqq qqqq qqqq
//   p 15     pp pppp pppp pppp p
//
// The maximum 46 bit result occurs 
// when the seed is at its highest
// allowable value: 0x7FFFFFFE.  
//
//    0x20D37FFF7CB2     
//
// which splits up like this    
//
//   q 31                        111 1111 1111 1111 0111 1100 1011 0010
//   p 15     10 0000 1101 0011 0
//          =  100 0001 1010 0110 
//
// In hex, these maxiumum values are:
//
//   q 31     7FFF7CB2  = 2^31 - (2 * 16807)
//   p 15         41A6  = 16807 - 1
//
//
// The task is to combine the two partial products p and q as if they were
// both parts of a 46 bit number, with the final result being modulo:
//
//                              0111 1111 1111 1111 1111 1111 1111 1111
//
// when we are actually only doing 32 bits at a time.  
//
// Here I explain David G. Carta's trick - in a different and much simpler 
// way than he does.
//
// We need to deal with the p bits "pp pppp pppp pppp p" shown above.  
// These bits carry weights of bits 45 to 31 in the multiplication product 
// of the usual Park Miller algorithm.
// 
// David Carta writes that in order to calculate mod(0x7FFFFFFF) of the
// complete multiplication product (taking into account the total value
// of p and q) we should simply add the bits of p into the bit positions 
// 14 to 0 of q and then do a mod(0x7FFFFFFF) on the result!  
//
//         |                   |                   |                   |
//          4444 4444 3333 3333 3322 2222 2222 1111 1111 11            
//          7654 3210 9876 5432 1098 7654 3210 9876 5432 1098 7654 3210
//
//     31                        qqq qqqq qqqq qqqq qqqq qqqq qqqq qqqq
//     15                   +                        ppp pppp pppp pppp
//                          =   Cxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
//                               
// Highest possible value,
// for q, with a value for
// p which would allow it:
//        
//                   7FFFFFFF    111 1111 1111 1111 1111 1111 1111 1111
//                +      41A5                        100 0001 1010 0101
//                = 8000041A4   1000 0000 0000 0000 0100 0001 1010 0100
//
// The result can't be larger than 2 * 0x7FFFFFFF = 0xFFFFFFFE.  So when we 
// do the modulus operation, we will have to subtract either nothing or just
// one 0x7FFFFFFF.  With this model of addition, the subtraction only 
// occurs very rarely.
//
// David Carta's explanation for why this produces the correct answer is too 
// long to repeat here.  Mine is easy to understand. 
//
// Lets define some labels:
//
//  Q = 31 bits 30 to 0. 
//  P = 15 bits 14 to 0.  
//
// If we were doing 46 bit math, the multiplication product (seed * 16807) 
// would be: 
//
//     Q
//  + (P * 0x80000000) 
//
// Observe that this is the same as:
//
//     Q
//  + (P * 0x7FFFFFFF) 
//  + (P * 0x00000001) 
//                                          
// However, we don't need or want a 46 bit result.  We only want that result
// mod(0x7FFFFFFF).  Therfore we can ignore the middle line above and use for 
// our result:
//  
//    Q
//  + P 
//
// This is a lot snappier than using a division, as the Schrage technique 
// requires.
//
// Methods:
//    
//    seedi    Set seed with a 31 bit unsigned integer between 1 and 
//             0x7FFFFFFE inclusive.  Don't use 0!
//
//    ranlui   Provides the next pseudorandom number as a long unsigned 
//             integer (31 bits).
//
//    ranf     Provides the next pseudorandom number as a float between
//             nearly 0 and nearly 1.0.

class rand31dc {
                                    // The sole item of state - a 32 bit 
                                    // integer.
    long unsigned int seed31;   

public:
                                    // Constructor sets seed31 to 1, in case 
                                    // no seedi operation is used.
    rand31dc() {seed31 = 1;}                                    
                                    
                                    // Declare methods.
                                    
    void              seedi(long unsigned int);
    long unsigned int ranlui(void);  
    float             ranf(void);
    

private:
                                    // nextrand()
                                    //
                                    // Generate next pseudo-random number.
                                    
                                    // Multiplier constant = 16807 = 7^5.  
                                    // This is 15 bits.
                                    //
                                    // Park and Miller in 1993 recommend
                                    // 48271, which they say produces a 
                                    // somewhat better quality of 
                                    // pseudo-random results.  
                                    //
                                    // 48271 can't be used with the 
                                    // following implementation of Carta's 
                                    // algorithm, since it is 16 bits and 
                                    // would result in bit 31 potentially 
                                    // being set in lo in the first
                                    // multiplication.  (A similar problem
                                    // occurs later with the higher bits of
                                    // hi.)

    #define consta 16807            
                                    // Modulus constant = 2^31 - 1 =
                                    // 0x7FFFFFFF.   We use this explicitly
                                    // in the code, rather than define it
                                    // somewhere, because this is a value
                                    // which must not be changed and should
                                    // always be recognised as a zero 
                                    // followed by 31 ones.
                                            
    long unsigned int nextrand()
    {
                                    // Two 32 bit integers for holding
                                    // parts of the (seed31 * consta)
                                    // multiplication product which would 
                                    // normally need a 46 bit word. 
                                    // 
                                    // lo 31 bits       30  -  0 
                                    // hi 30 bits   45  -  16  
                                    //
                                    // These overlap in their value.
                                    //
                                    // Bit 0 of hi has the same weight in 
                                    // the result as bit 16 of lo.
                                    
        long unsigned int hi, lo;

                                    // lo = 31 bits:
                                    //  
                                    //    low 16 bits (15-0) of seed31 
                                    //  * 15 bit consta 
                                     
        lo = consta * (seed31 & 0xFFFF);
        
                                    // hi = 30 bits:
                                    //
                                    //    high 15 bits (30-16) of seed31
                                    //  * 15 bit consta 
                                    
        hi = consta * (seed31 >> 16);
        
                                    // The new pseudo-random number is the 
                                    // 46 bit product mod(0x7FFFFFF).  Our
                                    // task is to calculate it with 32
                                    // bit words and maths, and without
                                    // division.
                                    //
                                    // The first section is easy to
                                    // understand.  We have a bunch of
                                    // bits - bits 14 to 0 of hi - 
                                    // which overlap with and carry the
                                    // same weight as bits 30 to 16 of
                                    // lo.
                                    //
                                    // Add the low 15 bits of hi into
                                    // bits 30-16 of lo.  
                                    
        lo += (hi & 0x7FFF) << 16;
        
                                    // The result may set bit 31 of lo, but
                                    // it will not overflow lo.
                                    //
                                    // So far, we got some of our total
                                    // result in lo.
                                    //
                                    // The only other part of the result
                                    // we need to deal with is bits
                                    // 29 to 15 of hi. 
                                    //
                                    // These bits carry weights of bits
                                    // 45 to 31 in the value of the 
                                    // multiplication product of the usual
                                    // Park-Miller algorithm.
                                    //
                                    // David Carta writes that in order
                                    // to get the mod(0x7FFFFFF) of the
                                    // multiplication product we should
                                    // simply add these bits into the
                                    // bit positions 14 to 0 of lo.
                                    //
        lo += hi >> 15;             //
                                    // In order to be able to get away with
                                    // this, and to perform the following
                                    // simple mod(0x7FFFFFFF) operation,
                                    // we need to be sure that the result 
                                    // of the addition will not exceed:
                                    //                                      
                                    // 2 * 0x7FFFFFFF = 0xFFFFFFFE
                                    // 
                                    // This is assured as per the diagrams
                                    // above.
                                    // Note that in the vast majority of 
                                    // cases, lo will be less than 
                                    // 0x7FFFFFFFF. 
                                    
        if (lo > 0x7FFFFFFF) lo -= 0x7FFFFFFF;          
        
                                    // lo contains the new pseudo-random
                                    // number.  Store it to the seed31 and
                                    // return it.
        
        return ( seed31 = (long)lo );       
    }
};

                                    /////////////////////////////////////////
                                    //
                                    // Implementations of the methods.
                                    
                                    // seedi()
                                    //
                                    // Set the seed from a long unsigned 
                                    // integer.  If zero is used, then
                                    // the seed will be set to 1.
                                                                        
void rand31dc::seedi(long unsigned int seedin)
{
    if (seedin == 0) seedin = 1;
    seed31 = seedin;
}
                                    
                                    // ranlui()
                                    //
                                    // Return next pseudo-random value as
                                    // a long unsigned integer.
                                    
long unsigned int rand31dc::ranlui(void)  
{
    return nextrand();
}

                                    // ranf()
                                    //
                                    // Return next pseudo-random value as
                                    // a floating point value.


float rand31dc::ranf(void)  
{
    return (nextrand() / 2147483647.0);
}

