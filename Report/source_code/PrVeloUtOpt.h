#pragma once

#include "Constants.h"
#include "DataStructures.h"
#include "Hlt1Track.h"
#include "UtUtil.h"

#include <tuple>
#include <utility>
#include <vector>


namespace opt {

class UtHitSpacePartition;


//------------------------------------------------------------------------------
// Output data of the velo ut algorithm == trash right now.
//------------------------------------------------------------------------------
using TrackObj = LHCb::HLT1::Track;
using PrVeloUtOutput = std::tuple<UtHits,
								  simd_vector<uint32_t>,
								  States,
								  std::vector<TrackObj>>;


//------------------------------------------------------------------------------
// Structs.
//------------------------------------------------------------------------------
struct SearchLayers {
	uint8_t base1, base2, extra1, extra2;
};

struct LineExtraHit {
	float minDiff;
	uint32_t index;
};


struct LinearSystem {
	std::array<float_v, 6> A;
	std::array<float_v, 3> b;
};


struct FittedTrackParameters {
	float_v xMidUt;
	float_v txMidUt;
	float_v yOffsetMidUt;
};


//------------------------------------------------------------------------------
// Constant parameters.
//------------------------------------------------------------------------------



//------------------------------------------------------------------------------
// Velo ut algorithm.
//------------------------------------------------------------------------------
class PrVeloUtOpt {
public:
	PrVeloUtOutput operator()(const std::vector<TrackObj>& veloTracks,
							  const States& veloEndStates,
							  const UtHitSpacePartition& hits) const;

public:
	using Alloc = StackAllocator<float>;

	// Utility
	static float_mask_v IsTrackInAcceptance(float_v x, float_v y, float_v z, float_v tx, float_v ty);
	inline static float_v GatherHitsTolerance(float_v trackTx, float_v trackTy);
	inline static LineExtraHit ScanExtraHits(const GatheredHitsS& hits, uint32_t beginIndex, uint32_t endIndex, float x0, float dxdz, float maxDiff);
	inline static std::array<int, 2> GetLineCombinatorics(const std::array<uint32_t, 4>& beginIndices, const std::array<uint32_t, 4>& endIndices);
	inline static std::pair<const SearchLayers&, const SearchLayers&> GetSearchConfig(int c1, int c2);
	inline static float_v EstimateKinkPosition(float_v ty);
	static std::array<float_v, 2> LinearRegressionFit(const UtTrackCandidatesS& candidates, uint32_t candidateIdx);
	static float_v EstimateQopInitial(float_v zKink, float_v xTrackKink, float_v x, float_v z, float_v tx, float_v ty, const UtTrackCandidatesS& candidates, uint32_t candidateIdx);
	static LinearSystem CreateFitLinearSystem(float_v zKink, float_v xTrackKink, float_v weightVelo, const UtTrackCandidatesS& candidates, uint32_t candidateIdx);
	inline static float_v CalculateChi2(float_v xTrackKink, const FittedTrackParameters& fitParams, float_v zKink, float_v weightVelo, const UtTrackCandidatesS& candidates, uint32_t candidateIdx);
	inline static float_v CalculateQop(const FittedTrackParameters& fitParams, float_v zKink, float_v xTrackKink, float_v x, float_v z, float_v ty);
	inline static float_v CalculateWeightVelo(float_v qop, float_v ty);
	inline static float_v CalculateLinearDiscriminant(float_v p, float_v pt, float_v chi2);
	inline static float_v CalculateQpxz2p(float_v x, float_v z, float_v tx, float_v ty);
	inline static float_mask_v MarkPotentialGhosts(float_v chi2, float_v qop, float_v tx, float_v ty);

	// Bulk functions
	static auto FilterStates(const std::vector<TrackObj>& veloTracks, const States& veloEndStates, const Alloc& alloc = {}) -> std::tuple<std::vector<TrackObj>, StatesS>;
	static ExtrapolatedStatesS ExtrapolateStates(const StatesS& states, float z, const Alloc& alloc = {});
	static StateBinLocationsS GetBinLocations(const UtHitSpacePartition& hits, const ExtrapolatedStatesS& layerStates, size_t& totalNumBins, const Alloc& alloc = {});
	static stack_vector<float> GetHitTolerances(const StatesS& states, const ExtrapolatedStatesS& layerStates, const Alloc& alloc = {});
	static std::pair<GatheredHitsS, stack_vector<uint32_t>> GatherHits(const UtHitSpacePartition& hits, const StatesS& states, const ExtrapolatedStatesS& layerStates, const Alloc& alloc = {});
	static UtTrackCandidatesS FindLines(const std::array<GatheredHitsS, 4>& gatheredHits, const std::array<stack_vector<uint32_t>, 4>& gatheredOffests, const StatesS& states, const Alloc& alloc = {});
	static uint32_t FindLinesForTrack(const std::array<GatheredHitsS, 4>& gatheredHits,
									  const std::array<uint32_t, 4>& beginIndices,
									  const std::array<uint32_t, 4>& endIndices,
									  SearchLayers layerConfig,
									  UtTrackCandidatesS& outCandidates,
									  size_t writeIndex);
	static FittedTrackPropertiesS FitTrackCandidates(const UtTrackCandidatesS& candidates, const Alloc& alloc = {});
	static stack_vector<float> ScoreCandidates(const UtTrackCandidatesS& candidates, const FittedTrackPropertiesS& fit, const Alloc& alloc = {});
	static stack_vector<uint32_t> CandidateRunLengths(const UtTrackCandidatesS& candidates, const Alloc& alloc = {});
	static stack_vector<uint32_t> SelectBestCandidates(const UtTrackCandidatesS& candidates, const stack_vector<float>& scores, const Alloc& alloc = {});
	static AssembledTracks AssembleTracks(const stack_vector<uint32_t>& selectedCandidates, const UtTrackCandidatesS& candidates, const FittedTrackPropertiesS& fitResults);

protected:
	static constexpr float sineFiberAngles[4] = { 0.0f, -0.08715f, 0.08715f, 0.0f };
	static constexpr float zUnitScaling = 0.001f;
	static constexpr std::array<float, 3> magFieldParams = { 2010.0f, -2240.0f, -71330.f };
	static constexpr float adaptiveToleranceConstant = 83.f; // This is a magic number obtained from old UT. Roughly 1/500*distToMomentum*minPt, I don't know what it means or if it's needed.
	static constexpr float maxChi2 = 1280.f;
	static constexpr float minMomentum = 2500; // MeV
	static constexpr float minTransverseMomentum = 425; // MeV
	static constexpr float minDiscriminant = -0.5f;
	inline static const std::array<SearchLayers, 2> lineSearchConfigs = {
		SearchLayers{ 0, 2, 1, 3 },
		SearchLayers{ 1, 3, 0, 2 },
	};
};



//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------
inline float_v PrVeloUtOpt::GatherHitsTolerance(float_v trackTx, float_v trackTy) {
	const float_v adaptiveTolerance = adaptiveToleranceConstant * Vc::sqrt(trackTx * trackTx + trackTy * trackTy);
	return Vc::min(float_v(16.0f), Vc::max(float_v(0.2f), adaptiveTolerance));
}


inline LineExtraHit PrVeloUtOpt::ScanExtraHits(const GatheredHitsS& hits, uint32_t beginIndex, uint32_t endIndex, float x0, float dxdz, float maxDiff) {
	float minDiff = maxDiff;
	uint32_t index = 0;
	for (size_t hitIdx = beginIndex; hitIdx < endIndex; ++hitIdx) {
		const float xe = hits.x[hitIdx];
		const float ze = hits.z[hitIdx];
		const float xexp = x0 + dxdz * ze;
		const float xdiff = std::abs(xexp - xe);
		index = xdiff < minDiff ? hitIdx : index;
		minDiff = xdiff < minDiff ? xdiff : minDiff;
	}
	return { minDiff, index };
}


inline std::array<int, 2> PrVeloUtOpt::GetLineCombinatorics(const std::array<uint32_t, 4>& beginIndices, const std::array<uint32_t, 4>& endIndices) {
	const std::array<uint32_t, 4> counts = {
		endIndices[0] - beginIndices[0],
		endIndices[1] - beginIndices[1],
		endIndices[2] - beginIndices[2],
		endIndices[3] - beginIndices[3],
	};

	const int combinatorics1 = counts[lineSearchConfigs[0].base1]
							   * counts[lineSearchConfigs[0].base2]
							   * (counts[lineSearchConfigs[0].extra1] + counts[lineSearchConfigs[0].extra2]);

	const int combinatorics2 = counts[lineSearchConfigs[1].base1]
							   * counts[lineSearchConfigs[1].base2]
							   * (counts[lineSearchConfigs[1].extra1] + counts[lineSearchConfigs[1].extra2]);

	return { combinatorics1, combinatorics2 };
}


inline std::pair<const SearchLayers&, const SearchLayers&> PrVeloUtOpt::GetSearchConfig(int c1, int c2) {
	using Ret = std::pair<const SearchLayers&, const SearchLayers&>;
	return c1 > c2 ? Ret{ lineSearchConfigs[0], lineSearchConfigs[1] } : Ret{ lineSearchConfigs[1], lineSearchConfigs[0] };
}


inline float_v PrVeloUtOpt::EstimateKinkPosition(float_v ty) {
	float_v tyPow2 = ty * ty;
	float_v tyPow4 = tyPow2 * tyPow2;
	return magFieldParams[0] - tyPow2 * magFieldParams[1] - tyPow4 * magFieldParams[2];
}


inline float_v PrVeloUtOpt::CalculateChi2(float_v xTrackKink, const FittedTrackParameters& fitParams, float_v zKink, float_v weightVelo, const UtTrackCandidatesS& candidates, uint32_t candidateIdx) {
	const float_v ty = SimdElement(candidates.trackTy, candidateIdx);

	const float_v distX = xTrackKink - fitParams.xMidUt - fitParams.txMidUt * (zKink - float_v(zMidUT));
	const float_v distCorrectionX2 = reciprocal(1 + fitParams.txMidUt * fitParams.txMidUt);
	float_v chi2 = weightVelo * (distX * distX * distCorrectionX2 + (fitParams.yOffsetMidUt * fitParams.yOffsetMidUt) / (1.0f + ty * ty));

	for (int layer = 0; layer < 4; ++layer) {
		const float_v w = SimdElement(candidates.hitWeight[layer], candidateIdx);
		const float_v dz = SimdElement(candidates.hitZ[layer], candidateIdx) - float_v(zMidUT);
		const float_v hitX = SimdElement(candidates.hitAccurateX[layer], candidateIdx);
		const float_v dist = hitX - fitParams.xMidUt - fitParams.txMidUt * dz - fitParams.yOffsetMidUt * float_v(sineFiberAngles[layer]);
		chi2 += w * dist * dist * distCorrectionX2;
	}

	return chi2;
}


inline float_v PrVeloUtOpt::CalculateQop(const FittedTrackParameters& fitParams, float_v zKink, float_v xTrackKink, float_v x, float_v z, float_v ty) {
	const float_v xb = 0.5f * ((fitParams.xMidUt + fitParams.txMidUt * (zKink - zMidUT)) + xTrackKink); // the 0.5 is empirical
	const float_v xSlopeVeloFit = (xb - x) / (zKink - z);

	const float_v sinInX = xSlopeVeloFit * rsqrt(1.0f + xSlopeVeloFit * xSlopeVeloFit + ty * ty);
	const float_v sinOutX = fitParams.txMidUt * rsqrt(1.0f + fitParams.txMidUt * fitParams.txMidUt + ty * ty);
	const float_v qop = (sinInX - sinOutX);

	return qop; // multiply by qopxz2p
}


inline float_v PrVeloUtOpt::CalculateWeightVelo(float_v qop, float_v ty) {
	const float_v invP = Vc::abs(qop * rsqrt(1.0f + ty * ty));
	const float_v multipleScatteringError = 0.14f + 10000.0f * invP;
	const float_v veloResolutionError = 0.12f + 3000.0f * invP;
	const float_v totalError = multipleScatteringError * multipleScatteringError + veloResolutionError * veloResolutionError;
	return reciprocal(totalError);
}


inline float_v PrVeloUtOpt::CalculateLinearDiscriminant(float_v p, float_v pt, float_v chi2) {
	constexpr float coeffs[4] = { 0.1989454476255, -0.10045676719805, 0.1224884024035, -0.154115981628 };
	float_v discriminant = coeffs[0]
						   + coeffs[1] * Vc::log(p)
						   + coeffs[2] * Vc::log(pt)
						   + coeffs[3] * Vc::log(chi2);
	return discriminant;
}


inline float_v PrVeloUtOpt::CalculateQpxz2p(float_v x, float_v z, float_v tx, float_v ty) {
	const float_v zOrigin = z - x / tx;
	const float_v bdl = IntegratedMagneticField770(ty, zOrigin);
	const float_v qpxz2p = -3.3356e-3f * reciprocal(bdl);
	return qpxz2p;
}


inline float_mask_v PrVeloUtOpt::MarkPotentialGhosts(float_v chi2, float_v qop, float_v tx, float_v ty) {
	const float_v momentum = abs(reciprocal(qop));
	const float_v transverseMomentum = momentum * Vc::sqrt(tx * tx + ty * ty);
	const float_mask_v momentumGhostMask = momentum < minMomentum || transverseMomentum < minTransverseMomentum;

	const float_v discriminant = CalculateLinearDiscriminant(momentum, transverseMomentum, chi2);
	const float_mask_v discriminantGhostMask = discriminant < minDiscriminant;

	return momentumGhostMask || discriminantGhostMask;
}


inline int PopCount(int bitset) {
#ifdef _MSC_VER
	return __popcnt(bitset);
#else
	return __builtin_popcount(bitset);
#endif
}


} // namespace opt