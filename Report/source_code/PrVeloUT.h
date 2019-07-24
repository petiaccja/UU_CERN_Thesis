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
#pragma once

// Include files
//
#include <numeric>

// from Gaudi
#include "GaudiAlg/ISequencerTimerTool.h"
#include "GaudiAlg/Transformer.h"

// from TrackInterfaces
#include "Event/RecVertex_v2.h"
#include "Event/Track_v2.h"

#include "PrKernel/UTHitHandler.h"

#include "GaudiKernel/DataObject.h"
#include "GaudiKernel/ObjectContainerBase.h"
#include "GaudiKernel/Range.h"
#include "Kernel/ILHCbMagnetSvc.h"
#include "PrKernel/UTHit.h"
#include "PrKernel/UTHitInfo.h"
#include "PrUTMagnetTool.h"
#include "TfKernel/IndexedHitContainer.h"
#include "TfKernel/MultiIndexedHitContainer.h"

#include "PrKernel/PrVeloUTTrack.h"
#include "UTDAQ/UTDAQHelper.h"
#include "vdt/log.h"
#include "vdt/sqrt.h"

#include "Event/PrUpstreamTracks.h"
#include "Event/PrVeloTracks.h"

/** @class PrVeloUT PrVeloUT.h
 *
 *  PrVeloUT algorithm. This is just a wrapper,
 *  the actual pattern recognition is done in the 'PrVeloUTTool'.
 *
 *  - InputTracksName: Input location for Velo tracks
 *  - OutputTracksName: Output location for VeloTT tracks
 *  - TimingMeasurement: Do a timing measurement?
 *
 *  @author Mariusz Witek
 *  @date   2007-05-08
 *  @update for A-Team framework 2007-08-20 SHM
 *
 *  2017-03-01: Christoph Hasse (adapt to future framework)
 *  2019-04-26: Arthur Hennequin (change data Input/Output)
 */

struct MiniState final {
  float x, y, z, tx, ty;
};

struct TrackHelper final {
  TrackHelper( const MiniState& miniState, const float zKink, const float sigmaVeloSlope, const float maxPseudoChi2 )
      : state( miniState ), bestParams{{0.0f, maxPseudoChi2, 0.0f, 0.0f}} {
    xMidField       = state.x + state.tx * ( zKink - state.z );
    const float a   = sigmaVeloSlope * ( zKink - state.z );
    wb              = 1.0f / ( a * a );
    invKinkVeloDist = 1.0f / ( zKink - state.z );
  }

  MiniState                          state;
  std::array<const UT::Mut::Hit*, 4> bestHits = {nullptr, nullptr, nullptr, nullptr};
  std::array<float, 4>               bestParams;
  float                              wb, invKinkVeloDist, xMidField;
};

class PrVeloUT : public Gaudi::Functional::Transformer<LHCb::Pr::Upstream::Tracks( const LHCb::Pr::Velo::Tracks&,
                                                                                   const UT::HitHandler& )> {
public:
  /// Standard constructor
  PrVeloUT( const std::string& name, ISvcLocator* pSvcLocator );

  StatusCode initialize() override;

  LHCb::Pr::Upstream::Tracks operator()( const LHCb::Pr::Velo::Tracks&, const UT::HitHandler& ) const override final;

private:
  Gaudi::Property<float> m_minMomentum{this, "minMomentum", 1.5 * Gaudi::Units::GeV};
  Gaudi::Property<float> m_minPT{this, "minPT", 0.3 * Gaudi::Units::GeV};
  Gaudi::Property<float> m_minMomentumFinal{this, "minMomentumFinal", 2.5 * Gaudi::Units::GeV};
  Gaudi::Property<float> m_minPTFinal{this, "minPTFinal", 0.425 * Gaudi::Units::GeV};
  Gaudi::Property<float> m_maxPseudoChi2{this, "maxPseudoChi2", 1280.};
  Gaudi::Property<float> m_yTol{this, "YTolerance", 0.5 * Gaudi::Units::mm}; // 0.8
  Gaudi::Property<float> m_yTolSlope{this, "YTolSlope", 0.08};               // 0.2
  Gaudi::Property<float> m_hitTol1{this, "HitTol1", 6.0 * Gaudi::Units::mm};
  Gaudi::Property<float> m_hitTol2{this, "HitTol2", 0.8 * Gaudi::Units::mm}; // 0.8
  Gaudi::Property<float> m_deltaTx1{this, "DeltaTx1", 0.035};
  Gaudi::Property<float> m_deltaTx2{this, "DeltaTx2", 0.018}; // 0.02
  Gaudi::Property<float> m_maxXSlope{this, "MaxXSlope", 0.350};
  Gaudi::Property<float> m_maxYSlope{this, "MaxYSlope", 0.300};
  Gaudi::Property<float> m_centralHoleSize{this, "centralHoleSize", 33. * Gaudi::Units::mm};
  Gaudi::Property<float> m_intraLayerDist{this, "IntraLayerDist", 15.0 * Gaudi::Units::mm};
  Gaudi::Property<float> m_overlapTol{this, "OverlapTol", 0.7 * Gaudi::Units::mm};
  Gaudi::Property<float> m_passHoleSize{this, "PassHoleSize", 40. * Gaudi::Units::mm};
  Gaudi::Property<float> m_LD3Hits{this, "LD3HitsMin", -0.5};
  Gaudi::Property<float> m_LD4Hits{this, "LD4HitsMin", -0.5};

  // Gaudi::Property<int>   m_minHighThres   {this, "MinHighThreshold", 1}; // commented, as the threshold bit might /
  // will be removed
  Gaudi::Property<bool> m_printVariables{this, "PrintVariables", false};
  Gaudi::Property<bool> m_passTracks{this, "PassTracks", false};
  Gaudi::Property<bool> m_doTiming{this, "TimingMeasurement", false};
  Gaudi::Property<bool> m_finalFit{this, "FinalFit", true};
  Gaudi::Property<bool> m_fiducialCuts{this, "FiducialCuts", true};

  mutable Gaudi::Accumulators::SummingCounter<unsigned int> m_seedsCounter{this, "#seeds"};
  mutable Gaudi::Accumulators::SummingCounter<unsigned int> m_tracksCounter{this, "#tracks"};

  StatusCode recomputeGeometry();

  template <typename dType>
  bool getState( const LHCb::Pr::Velo::Tracks& inputTracks, int at, MiniState& trState,
                 LHCb::Pr::Upstream::Tracks& outputTracks ) const;

  template <typename FudgeTable>
  bool getHits( LHCb::span<UT::Mut::Hits, 4> hitsInLayers, const UT::HitHandler& hh, const FudgeTable& fudgeFactors,
                MiniState& trState ) const;

  bool formClusters( LHCb::span<const UT::Mut::Hits, 4> hitsInLayers, TrackHelper& helper ) const;

  template <typename dType, typename BdlTable>
  void prepareOutputTrack( const LHCb::Pr::Velo::Tracks& inputTracks, int ancestor, const TrackHelper& helper,
                           LHCb::span<const UT::Mut::Hits, 4> hitsInLayers, LHCb::Pr::Upstream::Tracks& outputTracks,
                           const BdlTable& bdlTable ) const;

  // ==============================================================================
  // -- Method that finds the hits in a given layer within a certain range
  // ==============================================================================
  template <typename RANGE>
  void findHits( RANGE range, float zInit, const MiniState& myState, const float xTolNormFact, const float invNormFact,
                 UT::Mut::Hits& hits ) const {

    const auto yApprox       = myState.y + myState.ty * ( zInit - myState.z );
    const auto xOnTrackProto = myState.x + myState.tx * ( zInit - myState.z );
    const auto yyProto       = myState.y - myState.ty * myState.z;

    auto first = std::find_if_not( range.begin(), range.end(),
                                   [y = m_yTol + m_yTolSlope * std::abs( xTolNormFact ), yApprox]( const auto& h ) {
                                     return h.isNotYCompatible( yApprox, y );
                                   } );
    for ( auto last = range.end(); first != last; ++first ) {
      const auto& hit = *first;

      const auto xx = hit.xAt( yApprox );
      const auto dx = xx - xOnTrackProto;

      if ( dx < -xTolNormFact ) continue;
      if ( dx > xTolNormFact ) break;

      // -- Now refine the tolerance in Y
      if ( hit.isNotYCompatible( yApprox, m_yTol + m_yTolSlope * std::abs( dx * invNormFact ) ) ) continue;

      const auto zz  = hit.zAtYEq0();
      const auto yy  = yyProto + myState.ty * zz;
      const auto xx2 = hit.xAt( yy );
      hits.emplace_back( &hit, xx2, zz );
    }
  }

  // ===========================================================================================
  // -- 2 helper functions for fit
  // -- Pseudo chi2 fit, templated for 3 or 4 hits
  // ===========================================================================================
  void addHit( LHCb::span<float, 3> mat, LHCb::span<float, 2> rhs, const UT::Mut::Hit& hit ) const {
    const float ui = hit.x;
    const float ci = hit.HitPtr->cosT();
    const float dz = 0.001f * ( hit.z - m_zMidUT );
    const float wi = hit.HitPtr->weight();
    mat[0] += wi * ci;
    mat[1] += wi * ci * dz;
    mat[2] += wi * ci * dz * dz;
    rhs[0] += wi * ui;
    rhs[1] += wi * ui * dz;
  }

  template <std::size_t N>
  void simpleFit( std::array<const UT::Mut::Hit*, N> hits, TrackHelper& helper ) const {
    static_assert( N == 3 || N == 4 );

    // commented, as the threshold bit might / will be removed
    // -- Veto hit combinations with no high threshold hit
    // -- = likely spillover
    // const int nHighThres = std::count_if( hits.begin(),  hits.end(),
    //                                      []( const UT::Mut::Hit* hit ){ return hit && hit->HitPtr->highThreshold();
    //                                      });

    // if( nHighThres < m_minHighThres ) return;

    // -- Scale the z-component, to not run into numerical problems
    // -- with floats
    const float zDiff = 0.001f * ( m_zKink - m_zMidUT );
    auto        mat   = std::array{helper.wb, helper.wb * zDiff, helper.wb * zDiff * zDiff};
    auto        rhs   = std::array{helper.wb * helper.xMidField, helper.wb * helper.xMidField * zDiff};
    std::for_each( hits.begin(), hits.end(), [&]( const auto* h ) { this->addHit( mat, rhs, *h ); } );

    ROOT::Math::CholeskyDecomp<float, 2> decomp( mat.data() );
    if ( UNLIKELY( !decomp ) ) return;

    decomp.Solve( rhs );

    const float xSlopeTTFit = 0.001f * rhs[1];
    const float xTTFit      = rhs[0];

    // new VELO slope x
    const float xb            = xTTFit + xSlopeTTFit * ( m_zKink - m_zMidUT );
    const float xSlopeVeloFit = ( xb - helper.state.x ) * helper.invKinkVeloDist;
    const float chi2VeloSlope = ( helper.state.tx - xSlopeVeloFit ) * m_invSigmaVeloSlope;

    const float chi2TT = std::accumulate( hits.begin(), hits.end(), chi2VeloSlope * chi2VeloSlope,
                                          [&]( float chi2, const auto* hit ) {
                                            const float du = ( xTTFit + xSlopeTTFit * ( hit->z - m_zMidUT ) ) - hit->x;
                                            return chi2 + hit->HitPtr->weight() * ( du * du );
                                          } ) /
                         ( N + 1 - 2 );

    if ( chi2TT < helper.bestParams[1] ) {

      // calculate q/p
      const float sinInX  = xSlopeVeloFit * vdt::fast_isqrtf( 1.0f + xSlopeVeloFit * xSlopeVeloFit );
      const float sinOutX = xSlopeTTFit * vdt::fast_isqrtf( 1.0f + xSlopeTTFit * xSlopeTTFit );
      const float qp      = ( sinInX - sinOutX );

      helper.bestParams = {qp, chi2TT, xTTFit, xSlopeTTFit};

      std::copy( hits.begin(), hits.end(), helper.bestHits.begin() );
      if ( N == 3 ) { helper.bestHits[3] = nullptr; }
    }
  }
  // --

  DeUTDetector* m_utDet = nullptr;

  /// Multipupose tool for Bdl and deflection
  ToolHandle<PrUTMagnetTool> m_PrUTMagnetTool{this, "PrUTMagnetTool", "PrUTMagnetTool"};
  ILHCbMagnetSvc*            m_magFieldSvc = nullptr;
  /// timing tool
  mutable ToolHandle<ISequencerTimerTool> m_timerTool{this, "SequencerTimerTool", "SequencerTimerTool"}; // FIXME
  ///< Counter for timing tool
  int m_veloUTTime{0};

  float m_zMidUT;
  float m_distToMomentum;
  float m_zKink{1780.0};
  float m_sigmaVeloSlope{0.10 * Gaudi::Units::mrad};
  float m_invSigmaVeloSlope{10.0 / Gaudi::Units::mrad};

  /// information about the different layers
  std::array<LHCb::UTDAQ::LayerInfo, 4>         m_layers;
  std::array<LHCb::UTDAQ::SectorsInStationZ, 2> m_sectorsZ;
};
