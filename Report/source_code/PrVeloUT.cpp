/*****************************************************************************\
* (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
// Include files
#include "Kernel/STLExtensions.h"
#include "UTDet/DeUTSector.h"
#include <boost/container/small_vector.hpp>

// local
#include "LHCbMath/GeomFun.h"
#include "PrVeloUT.h"

//-----------------------------------------------------------------------------
// Implementation file for class : PrVeloUT
//
// 2007-05-08: Mariusz Witek
// 2017-03-01: Christoph Hasse (adapt to future framework)
// 2019-04-26: Arthur Hennequin (change data Input/Output)
//-----------------------------------------------------------------------------

namespace {
  // -- parameters that describe the z position of the kink point as a function of ty in a 4th order polynomial (even
  // terms only)
  constexpr auto magFieldParams = std::array{2010.0f, -2240.0f, -71330.f};

  // perform a fit using trackhelper's best hits with y correction, improve qop estimate
  float fastfitter( const TrackHelper& helper, std::array<float, 4>& improvedParams, const float zMidUT,
                    const float qpxz2p ) {

    const float ty        = helper.state.ty;
    const float zKink     = magFieldParams[0] - ty * ty * magFieldParams[1] - ty * ty * ty * ty * magFieldParams[2];
    const float xMidField = helper.state.x + helper.state.tx * ( zKink - helper.state.z );

    const float zDiff = 0.001f * ( zKink - zMidUT );

    // -- This is to avoid division by zero...
    const float pHelper = std::max( float( std::abs( helper.bestParams[0] * qpxz2p ) ), float( 1e-9 ) );
    const float invP    = pHelper * vdt::fast_isqrtf( 1.0f + ty * ty );

    // these resolution are semi-empirical, could be tuned and might not be correct for low momentum.
    const float error1 =
        0.14f + 10000.0f * invP; // this is the resolution due to multiple scattering between Velo and UT
    const float error2 = 0.12f + 3000.0f * invP; // this is the resolution due to the finite Velo resolution
    const float error  = error1 * error1 + error2 * error2;
    const float weight = 1.0f / error;

    float mat[6] = {weight, weight * zDiff, weight * zDiff * zDiff, 0.0f, 0.0f, 0.0f};
    float rhs[3] = {weight * xMidField, weight * xMidField * zDiff, 0.0f};

    for ( auto hit : helper.bestHits ) {

      // -- only the last one can be a nullptr
      if ( hit == nullptr ) break;

      const float ui = hit->x;
      const float dz = 0.001f * ( hit->z - zMidUT );
      const float w  = hit->HitPtr->weight();
      const float t  = hit->HitPtr->sinT();

      mat[0] += w;
      mat[1] += w * dz;
      mat[2] += w * dz * dz;
      mat[3] += w * t;
      mat[4] += w * dz * t;
      mat[5] += w * t * t;
      rhs[0] += w * ui;
      rhs[1] += w * ui * dz;
      rhs[2] += w * ui * t;
    }

    ROOT::Math::CholeskyDecomp<float, 3> decomp( mat );
    if ( UNLIKELY( !decomp ) ) {
      return helper.bestParams[0];
    } else {
      decomp.Solve( rhs );
    }

    const float xSlopeUTFit = 0.001f * rhs[1];
    const float xUTFit      = rhs[0];
    const float offsetY     = rhs[2];

    const float distX = ( xMidField - xUTFit - xSlopeUTFit * ( zKink - zMidUT ) );
    // -- This takes into account that the distance between a point and track is smaller than the distance on the x-axis
    const float distCorrectionX2 = 1.0f / ( 1 + xSlopeUTFit * xSlopeUTFit );
    float       chi2 = weight * ( distX * distX * distCorrectionX2 + offsetY * offsetY / ( 1.0f + ty * ty ) );

    for ( auto hit : helper.bestHits ) {
      if ( hit == nullptr ) break;

      const float w    = hit->HitPtr->weight();
      const float dz   = hit->z - zMidUT;
      const float dist = ( hit->x - xUTFit - xSlopeUTFit * dz - offsetY * hit->HitPtr->sinT() );
      chi2 += w * dist * dist * distCorrectionX2;
    }

    // new VELO slope x
    const float xb = 0.5f * ( ( xUTFit + xSlopeUTFit * ( zKink - zMidUT ) ) + xMidField ); // the 0.5 is empirical
    const float xSlopeVeloFit = ( xb - helper.state.x ) / ( zKink - helper.state.z );

    improvedParams = {xUTFit, xSlopeUTFit, helper.state.y + helper.state.ty * ( zMidUT - helper.state.z ) + offsetY,
                      chi2};

    // calculate q/p
    const float sinInX  = xSlopeVeloFit * vdt::fast_isqrtf( 1.0f + xSlopeVeloFit * xSlopeVeloFit + ty * ty );
    const float sinOutX = xSlopeUTFit * vdt::fast_isqrtf( 1.0f + xSlopeUTFit * xSlopeUTFit + ty * ty );
    return ( sinInX - sinOutX );
  }

  // -- Evaluate the linear discriminant
  // -- Coefficients derived with LD method for p, pT and chi2 with TMVA
  template <std::size_t nHits>
  float evaluateLinearDiscriminant( const std::array<float, 3> inputValues ) {

    constexpr auto coeffs =
        ( nHits == 3 ? std::array{0.162880166064f, -0.107081172665f, 0.134153123662f, -0.137764853657f}
                     : std::array{0.235010729187f, -0.0938323617311f, 0.110823681145f, -0.170467109599f} );

    assert( coeffs.size() == inputValues.size() + 1 );
    return std::inner_product( std::next( coeffs.begin() ), coeffs.end(), inputValues.begin(), coeffs.front(),
                               std::plus<>{}, []( float c, float iv ) { return c * vdt::fast_logf( iv ); } );
  }

  // -- These things are all hardcopied from the PrTableForFunction
  // -- and PrUTMagnetTool
  // -- If the granularity or whatever changes, this will give wrong results

  int masterIndex( const int index1, const int index2, const int index3 ) {
    return ( index3 * 11 + index2 ) * 31 + index1;
  }

  constexpr auto minValsBdl = std::array{-0.3f, -250.0f, 0.0f};
  constexpr auto maxValsBdl = std::array{0.3f, 250.0f, 800.0f};
  constexpr auto deltaBdl   = std::array{0.02f, 50.0f, 80.0f};
  constexpr auto dxDyHelper = std::array{0.0f, 1.0f, -1.0f, 0.0f};
} // namespace

// Declaration of the Algorithm Factory
DECLARE_COMPONENT( PrVeloUT )

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
PrVeloUT::PrVeloUT( const std::string& name, ISvcLocator* pSvcLocator )
    : Transformer( name, pSvcLocator,
                   {KeyValue{"InputTracksName", "Rec/Track/Velo"}, KeyValue{"UTHits", UT::Info::HitLocation}},
                   KeyValue{"OutputTracksName", "Rec/Track/UT"} ) {}

/// Initialization
StatusCode PrVeloUT::initialize() {
  auto sc = Transformer::initialize();
  if ( sc.isFailure() ) return sc; // error printed already by GaudiAlgorithm

  // m_zMidUT is a position of normalization plane which should to be close to z middle of UT ( +- 5 cm ).
  // Cached once in PrVeloUTTool at initialization. No need to update with small UT movement.
  m_zMidUT = m_PrUTMagnetTool->zMidUT();
  // zMidField and distToMomentum is properly recalculated in PrUTMagnetTool when B field changes
  m_distToMomentum = m_PrUTMagnetTool->averageDist2mom();

  m_magFieldSvc = svc<ILHCbMagnetSvc>( "MagneticFieldSvc", true );

  if ( m_doTiming ) {
    m_timerTool->increaseIndent();
    m_veloUTTime = m_timerTool->addTimer( "Internal VeloUT Tracking" );
    m_timerTool->decreaseIndent();
  }

  // Get detector element.
  m_utDet = getDet<DeUTDetector>( DeUTDetLocation::UT );
  // Make sure we precompute z positions/sizes of the layers/sectors
  registerCondition( m_utDet->geometry(), &PrVeloUT::recomputeGeometry );

  return StatusCode::SUCCESS;
}

StatusCode PrVeloUT::recomputeGeometry() {
  LHCb::UTDAQ::computeGeometry( *m_utDet, m_layers, m_sectorsZ );
  return StatusCode::SUCCESS;
}

//=============================================================================
// Main execution
//=============================================================================
LHCb::Pr::Upstream::Tracks PrVeloUT::operator()( const LHCb::Pr::Velo::Tracks& inputTracks,
                                                 const UT::HitHandler&         hh ) const {
  if ( m_doTiming ) m_timerTool->start( m_veloUTTime );

  LHCb::Pr::Upstream::Tracks outputTracks;
  m_seedsCounter += inputTracks.size();

  const auto& fudgeFactors = m_PrUTMagnetTool->DxLayTable();
  const auto& bdlTable     = m_PrUTMagnetTool->BdlTable();

  std::array<UT::Mut::Hits, 4> hitsInLayers;
  for ( auto& it : hitsInLayers ) it.reserve( 8 ); // check this number!

  // for now only scalar, but with some adaptation it can be vectorized
  using dType = SIMDWrapper::scalar::types;

  for ( int t = 0; t < inputTracks.size(); t++ ) {
    MiniState trState;
    if ( !getState<dType>( inputTracks, t, trState, outputTracks ) ) continue;

    for ( auto& it : hitsInLayers ) it.clear();
    if ( !getHits( hitsInLayers, hh, fudgeFactors, trState ) ) continue;

    TrackHelper helper( trState, m_zKink, m_sigmaVeloSlope, m_maxPseudoChi2 );

    if ( !formClusters( hitsInLayers, helper ) ) {
      std::reverse( hitsInLayers.begin(), hitsInLayers.end() );
      formClusters( hitsInLayers, helper );
      std::reverse( hitsInLayers.begin(), hitsInLayers.end() );
    }

    if ( helper.bestHits[0] ) prepareOutputTrack<dType>( inputTracks, t, helper, hitsInLayers, outputTracks, bdlTable );
  }

  m_tracksCounter += outputTracks.size();
  if ( m_doTiming ) m_timerTool->stop( m_veloUTTime );
  return outputTracks;
}
//=============================================================================
// Get the state, do some cuts
//=============================================================================
template <typename dType>
bool PrVeloUT::getState( const LHCb::Pr::Velo::Tracks& inputTracks, int at, MiniState& trState,
                         LHCb::Pr::Upstream::Tracks& outputTracks ) const {
  using I = typename dType::int_v;
  using F = typename dType::float_v;

  const int EndVelo = 1;
  auto      pos     = inputTracks.statePos<F>( at, EndVelo );
  auto      dir     = inputTracks.stateDir<F>( at, EndVelo );
  auto      cov     = inputTracks.stateCov<F>( at, EndVelo );

  // -- reject tracks outside of acceptance or pointing to the beam pipe
  trState.tx = dir.x.cast();
  trState.ty = dir.y.cast();
  trState.x  = pos.x.cast();
  trState.y  = pos.y.cast();
  trState.z  = pos.z.cast();

  const float xMidUT = trState.x + trState.tx * ( m_zMidUT - trState.z );
  const float yMidUT = trState.y + trState.ty * ( m_zMidUT - trState.z );

  if ( xMidUT * xMidUT + yMidUT * yMidUT < m_centralHoleSize * m_centralHoleSize ) return false;
  if ( ( std::abs( trState.tx ) > m_maxXSlope ) || ( std::abs( trState.ty ) > m_maxYSlope ) ) return false;

  if ( m_passTracks && std::abs( xMidUT ) < m_passHoleSize && std::abs( yMidUT ) < m_passHoleSize ) {
    int i    = outputTracks.size();
    int mask = true; // dummy mask to be replace if we want to vectorize

    outputTracks.compressstore_trackVP<I>( i, mask, at ); // ancestor
    outputTracks.compressstore_statePos<F>( i, mask, pos );
    outputTracks.compressstore_stateDir<F>( i, mask, dir );
    outputTracks.compressstore_stateCov<F>( i, mask, cov );
    outputTracks.compressstore_stateQoP<F>( i, mask, 0.f ); // no momentum
    outputTracks.compressstore_nHits<I>( i, mask, 0 );      // no hits

    outputTracks.size() += dType::popcount( mask );

    return false;
  }

  return true;
}

//=============================================================================
// Find the hits
//=============================================================================
template <typename FudgeTable>
bool PrVeloUT::getHits( LHCb::span<UT::Mut::Hits, 4> hitsInLayers, const UT::HitHandler& hh,
                        const FudgeTable& fudgeFactors, MiniState& trState ) const {

  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float                absSlopeY = std::abs( trState.ty );
  const int                  index     = (int)( absSlopeY * 100 + 0.5f );
  LHCb::span<const float, 4> normFact{&fudgeFactors.table()[4 * index], 4};

  // -- this 500 seems a little odd...
  const float invTheta =
      std::min( 500.0f, 1.0f * vdt::fast_isqrtf( trState.tx * trState.tx + trState.ty * trState.ty ) );
  const float minMom = std::max( m_minPT.value() * invTheta, m_minMomentum.value() );
  const float xTol   = std::abs( 1.0f / ( m_distToMomentum * minMom ) );
  const float yTol   = m_yTol + m_yTolSlope * xTol;

  int                                                    nLayers = 0;
  boost::container::small_vector<std::pair<int, int>, 9> sectors;

  for ( int iStation = 0; iStation < 2; ++iStation ) {

    if ( iStation == 1 && nLayers == 0 ) { return false; }

    for ( int iLayer = 0; iLayer < 2; ++iLayer ) {
      if ( iStation == 1 && iLayer == 1 && nLayers < 2 ) return false;

      const unsigned int layerIndex  = 2 * iStation + iLayer;
      const float        z           = m_layers[layerIndex].z;
      const float        yAtZ        = trState.y + trState.ty * ( z - trState.z );
      const float        xLayer      = trState.x + trState.tx * ( z - trState.z );
      const float        yLayer      = yAtZ + yTol * m_layers[layerIndex].dxDy;
      const float        normFactNum = normFact[layerIndex];
      const float        invNormFact = 1.0f / normFactNum;

      LHCb::UTDAQ::findSectors( layerIndex, xLayer, yLayer,
                                xTol * invNormFact - std::abs( trState.tx ) * m_intraLayerDist.value(),
                                m_yTol + m_yTolSlope * std::abs( xTol * invNormFact ), m_layers[layerIndex], sectors );

      const LHCb::UTDAQ::SectorsInLayerZ& sectorsZForLayer = m_sectorsZ[iStation][iLayer];
      std::pair                           pp{-1, -1};
      for ( auto& p : sectors ) {
        // sectors can be duplicated in the list, but they are ordered
        if ( p == pp ) continue;
        pp                    = p;
        const int fullChanIdx = ( layerIndex * 3 + ( p.first - 1 ) ) * 98 + ( p.second - 1 );
        findHits( hh.hits( fullChanIdx ), sectorsZForLayer[p.first - 1][p.second - 1], trState, xTol * invNormFact,
                  invNormFact, hitsInLayers[layerIndex] );
      }
      sectors.clear();
      nLayers += int( !hitsInLayers[2 * iStation + iLayer].empty() );
    }
  }

  return nLayers > 2;
}

//=========================================================================
// Form clusters
//=========================================================================
bool PrVeloUT::formClusters( LHCb::span<const UT::Mut::Hits, 4> hitsInLayers, TrackHelper& helper ) const {

  bool fourLayerSolution = false;

  for ( const auto& hit0 : hitsInLayers[0] ) {

    const float xhitLayer0 = hit0.x;
    const float zhitLayer0 = hit0.z;

    // Loop over Second Layer
    for ( const auto& hit2 : hitsInLayers[2] ) {

      const float xhitLayer2 = hit2.x;
      const float zhitLayer2 = hit2.z;

      const float tx = ( xhitLayer2 - xhitLayer0 ) / ( zhitLayer2 - zhitLayer0 );

      if ( std::abs( tx - helper.state.tx ) > m_deltaTx2 ) continue;

      const UT::Mut::Hit* bestHit1 = nullptr;
      float               hitTol   = m_hitTol2;
      for ( auto& hit1 : hitsInLayers[1] ) {

        const float xhitLayer1 = hit1.x;
        const float zhitLayer1 = hit1.z;

        const float xextrapLayer1 = xhitLayer0 + tx * ( zhitLayer1 - zhitLayer0 );
        if ( std::abs( xhitLayer1 - xextrapLayer1 ) < hitTol ) {
          hitTol   = std::abs( xhitLayer1 - xextrapLayer1 );
          bestHit1 = &hit1;
        }
      }

      if ( fourLayerSolution && !bestHit1 ) continue;

      const UT::Mut::Hit* bestHit3 = nullptr;
      hitTol                       = m_hitTol2;
      for ( auto& hit3 : hitsInLayers[3] ) {

        const float xhitLayer3 = hit3.x;
        const float zhitLayer3 = hit3.z;

        const float xextrapLayer3 = xhitLayer2 + tx * ( zhitLayer3 - zhitLayer2 );

        if ( std::abs( xhitLayer3 - xextrapLayer3 ) < hitTol ) {
          hitTol   = std::abs( xhitLayer3 - xextrapLayer3 );
          bestHit3 = &hit3;
        }
      }

      // -- All hits found
      if ( bestHit1 && bestHit3 ) {
        simpleFit( std::array{&hit0, bestHit1, &hit2, bestHit3}, helper );

        if ( !fourLayerSolution && helper.bestHits[0] ) { fourLayerSolution = true; }
        continue;
      }

      // -- Nothing found in layer 3
      if ( !fourLayerSolution && bestHit1 ) {
        simpleFit( std::array{&hit0, bestHit1, &hit2}, helper );
        continue;
      }
      // -- Noting found in layer 1
      if ( !fourLayerSolution && bestHit3 ) {
        simpleFit( std::array{&hit0, bestHit3, &hit2}, helper );
        continue;
      }
    }
  }

  return fourLayerSolution;
}

//=========================================================================
// Create the Velo-UT tracks
//=========================================================================
template <typename dType, typename BdlTable>
void PrVeloUT::prepareOutputTrack( const LHCb::Pr::Velo::Tracks& inputTracks, int ancestor, const TrackHelper& helper,
                                   LHCb::span<const UT::Mut::Hits, 4> hitsInLayers,
                                   LHCb::Pr::Upstream::Tracks& outputTracks, const BdlTable& bdlTable ) const {
  using I = typename dType::int_v;
  using F = typename dType::float_v;

  //== Handle states. copy Velo one, add TT.
  const float zOrigin = ( std::fabs( helper.state.ty ) > 0.001f ) ? helper.state.z - helper.state.y / helper.state.ty
                                                                  : helper.state.z - helper.state.x / helper.state.tx;

  // const float bdl1    = m_PrUTMagnetTool->bdlIntegral(helper.state.ty,zOrigin,helper.state.z);

  // -- These are calculations, copied and simplified from PrTableForFunction
  // -- FIXME: these rely on the internal details of PrTableForFunction!!!
  //           and should at least be put back in there, and used from here
  //           to make sure everything _stays_ consistent...
  const auto var = std::array{helper.state.ty, zOrigin, helper.state.z};

  const int index1 = std::max( 0, std::min( 30, int( ( var[0] + 0.3f ) / 0.6f * 30 ) ) );
  const int index2 = std::max( 0, std::min( 10, int( ( var[1] + 250 ) / 500 * 10 ) ) );
  const int index3 = std::max( 0, std::min( 10, int( var[2] / 800 * 10 ) ) );

  float bdl = bdlTable.table()[masterIndex( index1, index2, index3 )];

  const auto bdls = std::array{bdlTable.table()[masterIndex( index1 + 1, index2, index3 )],
                               bdlTable.table()[masterIndex( index1, index2 + 1, index3 )],
                               bdlTable.table()[masterIndex( index1, index2, index3 + 1 )]};

  const auto boundaries = std::array{-0.3f + float( index1 ) * deltaBdl[0], -250.0f + float( index2 ) * deltaBdl[1],
                                     0.0f + float( index3 ) * deltaBdl[2]};

  // -- This is an interpolation, to get a bit more precision
  float addBdlVal = 0.0;
  for ( int i = 0; i < 3; ++i ) {

    if ( var[i] < minValsBdl[i] || var[i] > maxValsBdl[i] ) continue;

    const float dTab_dVar = ( bdls[i] - bdl ) / deltaBdl[i];
    const float dVar      = ( var[i] - boundaries[i] );
    addBdlVal += dTab_dVar * dVar;
  }
  bdl += addBdlVal;
  // ----

  // -- order is: x, tx, y, chi2
  std::array<float, 4> finalParams = {helper.bestParams[2], helper.bestParams[3],
                                      helper.state.y + helper.state.ty * ( m_zMidUT - helper.state.z ),
                                      helper.bestParams[1]};

  const float qpxz2p = -1.0f / bdl * 3.3356f / Gaudi::Units::GeV;
  const float qp     = m_finalFit ? fastfitter( helper, finalParams, m_zMidUT, qpxz2p )
                              : helper.bestParams[0] * vdt::fast_isqrtf( 1.0f + helper.state.ty * helper.state.ty );
  const float qop = ( std::abs( bdl ) < 1.e-8f ) ? 0.0f : qp * qpxz2p;

  // -- Don't make tracks that have grossly too low momentum
  // -- Beware of the momentum resolution!
  const float p  = std::abs( 1.0f / qop );
  const float pt = p * std::sqrt( helper.state.tx * helper.state.tx + helper.state.ty * helper.state.ty );

  if ( p < m_minMomentumFinal || pt < m_minPTFinal ) return;

  const float xUT  = finalParams[0];
  const float txUT = finalParams[1];
  const float yUT  = finalParams[2];

  // -- apply some fiducial cuts
  // -- they are optimised for high pT tracks (> 500 MeV)
  if ( m_fiducialCuts ) {
    const float magSign = m_magFieldSvc->signedRelativeCurrent();

    if ( magSign * qop < 0.0f && xUT > -48.0f && xUT < 0.0f && std::abs( yUT ) < 33.0f ) return;
    if ( magSign * qop > 0.0f && xUT < 48.0f && xUT > 0.0f && std::abs( yUT ) < 33.0f ) return;

    if ( magSign * qop < 0.0f && txUT > 0.09f + 0.0003f * pt ) return;
    if ( magSign * qop > 0.0f && txUT < -0.09f - 0.0003f * pt ) return;
  }

  // -- evaluate the linear discriminant and reject ghosts
  // -- the values only make sense if the final fit is performed
  if ( m_finalFit ) {
    const auto nHits = std::count_if( helper.bestHits.begin(), helper.bestHits.end(), []( auto hit ) { return hit; } );
    if ( nHits == 3 ) {
      if ( evaluateLinearDiscriminant<3>( {p, pt, finalParams[3]} ) < m_LD3Hits ) return;
    } else {
      if ( evaluateLinearDiscriminant<4>( {p, pt, finalParams[3]} ) < m_LD4Hits ) return;
    }
  }

  // Make tracks :
  int i    = outputTracks.size();
  int mask = true; // dummy mask

  // Refined state ?
  // auto pos = Vec3<F>( helper.state.x, helper.state.y, helper.state.z );
  // auto dir = Vec3<F>( helper.state.tx, helper.state.ty, 1.f );

  // Or EndVelo state ?
  auto pos = inputTracks.statePos<F>( ancestor, 1 );
  auto dir = inputTracks.stateDir<F>( ancestor, 1 );
  auto cov = inputTracks.stateCov<F>( ancestor, 1 );

  outputTracks.compressstore_trackVP<I>( i, mask, ancestor );
  outputTracks.compressstore_statePos<F>( i, mask, pos );
  outputTracks.compressstore_stateDir<F>( i, mask, dir );
  outputTracks.compressstore_stateCov<F>( i, mask, cov );
  outputTracks.compressstore_stateQoP<F>( i, mask, qop );

  int n_hits = 0;
  for ( const auto* hit : helper.bestHits ) {
    if ( !hit ) break; // only the last one can be a nullptr.

    outputTracks.compressstore_hit<I>( i, n_hits, mask, (int)hit->HitPtr->lhcbID().channelID() );
    n_hits++;

    const float xhit = hit->x;
    const float zhit = hit->z;

    for ( auto& ohit : hitsInLayers[hit->HitPtr->planeCode()] ) {
      const float zohit = ohit.z;
      if ( zohit == zhit ) continue;

      const float xohit   = ohit.x;
      const float xextrap = xhit + txUT * ( zohit - zhit );
      if ( xohit - xextrap < -m_overlapTol ) continue;
      if ( xohit - xextrap > m_overlapTol ) break;

      if ( n_hits > 30 ) continue;
      outputTracks.compressstore_hit<I>( i, n_hits, mask, (int)ohit.HitPtr->lhcbID().channelID() );
      n_hits++;

      // only one overlap hit
      // break;
    }
  }
  outputTracks.compressstore_nHits<I>( i, mask, n_hits );

  outputTracks.size() += dType::popcount( mask );
}
