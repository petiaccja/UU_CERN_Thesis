#include "PrVeloUtOpt.h"

#include "Constants.h"
#include "UtHitSpacePartition.h"

#include <chrono>
#include <future>
#include <numeric>
#ifdef __GNUC__
#include <fenv.h>
#endif

namespace opt {


PrVeloUtOutput PrVeloUtOpt::operator()(const std::vector<TrackObj>& veloTracks,
									   const States& veloEndStates,
									   const UtHitSpacePartition& hits) const {
// FPEs don't work with AVX2 as it reads memory garbage intentionally, so save environment and disable FPEs.
#ifdef __GNUC__
	int fpeMask = fegetexcept();
	fedisableexcept(fpeMask);
#endif

	StackAllocator<float> alloc(512*1024, 32);

	const auto [filteredTracks, filteredStates] = FilterStates(veloTracks, veloEndStates, alloc);

	std::array<ExtrapolatedStatesS, 4> layerStates = {
		ExtrapolateStates(filteredStates, layersInfos[0].z, alloc),
		ExtrapolateStates(filteredStates, layersInfos[1].z, alloc),
		ExtrapolateStates(filteredStates, layersInfos[2].z, alloc),
		ExtrapolateStates(filteredStates, layersInfos[3].z, alloc),
	};

	std::array<GatheredHitsS, 4> gatheredHits;
	std::array<stack_vector<uint32_t>, 4> gatheredOffests;

	for (size_t i = 0; i < 4; ++i) {
		auto [h, o] = GatherHits(hits, filteredStates, layerStates[i], alloc);
		gatheredHits[i] = std::move(h);
		gatheredOffests[i] = std::move(o);
	}

	auto trackCandidates = FindLines(gatheredHits, gatheredOffests, filteredStates, alloc);
	auto fitResults = FitTrackCandidates(trackCandidates, alloc);
	auto scores = ScoreCandidates(trackCandidates, fitResults, alloc);
	auto bestCandidates = SelectBestCandidates(trackCandidates, scores, alloc);
	auto assembledTracks = AssembleTracks(bestCandidates, trackCandidates, fitResults);

	// Ugly hacky code to produce acceptable output.
	// TODO: refactor to proper format and function.
	const auto numTracks = assembledTracks.trackIndex.size();

	UtHits finalHits;
	simd_vector<uint32_t> finalHitOffsets;
	States finalStates;
	std::vector<TrackObj> finalTracks;
	finalHits.resize(4 * numTracks);
	finalHitOffsets.resize(numTracks);
	finalStates.resize(numTracks);
	finalTracks.resize(numTracks);


	uint32_t hitOffset = 0;
	for (size_t i = 0; i < assembledTracks.size(); ++i) {
		finalHitOffsets[i] = hitOffset;
		for (int layer = 0; layer < 4; ++layer) {
			const auto index = assembledTracks.hitIndices[layer][i];
			if (index != uint32_t(-1)) {
				CopyUtHit(hits.GetPartitionedHits(), index, finalHits, hitOffset);
				++hitOffset;
			}
		}
		CopyState(filteredStates, assembledTracks.trackIndex[i], finalStates, i);
		finalStates.qop[i] = assembledTracks.qop[i];
		finalTracks[i] = filteredTracks[assembledTracks.trackIndex[i]];
	}
	finalHits.resize(hitOffset);

	// Reset FPE environment to its previous state.
#ifdef __GNUC__
	feclearexcept(fpeMask);
	feenableexcept(fpeMask);
#endif

	return { std::move(finalHits), std::move(finalHitOffsets), std::move(finalStates), std::move(finalTracks) };
}


float_mask_v PrVeloUtOpt::IsTrackInAcceptance(float_v x, float_v y, float_v z, float_v tx, float_v ty) {
	static const float_v z1 = layersInfos[0].z;
	static const float_v z2 = layersInfos[3].z;

	const float_v x1 = x + tx * (z1 - z);
	const float_v x2 = x + tx * (z2 - z);
	const float_v y1 = y + ty * (z1 - z);
	const float_v y2 = y + ty * (z2 - z);

	const float_mask_v withinFirstLayer = detectorLeft < x1 && x1 < detectorRight && detectorBottom < y1 && y1 < detectorTop;
	const float_mask_v withinLastLayer = detectorLeft < x2 && x2 < detectorRight && detectorBottom < y2 && y2 < detectorTop;

	return withinFirstLayer && withinLastLayer;
}


auto PrVeloUtOpt::FilterStates(const std::vector<TrackObj>& veloTracks, const States& veloEndStates, const Alloc& alloc) -> std::tuple<std::vector<TrackObj>, StatesS> {
	assert(veloTracks.size() == veloEndStates.size());

	std::vector<TrackObj> filteredTracks;
	StatesS filteredStates(alloc);
	filteredTracks.resize(veloTracks.size());
	filteredStates.resize(veloEndStates.size());

	size_t size = veloEndStates.size();
	size_t writeIndex = 0;

	for (size_t i = 0; i < size; i += SimdWidth) {
		const float_v x = SimdElement(veloEndStates.x, i);
		const float_v y = SimdElement(veloEndStates.y, i);
		const float_v z = SimdElement(veloEndStates.z, i);
		const float_v tx = SimdElement(veloEndStates.tx, i);
		const float_v ty = SimdElement(veloEndStates.ty, i);

		const float_mask_v withinAcceptance = IsTrackInAcceptance(x, y, z, tx, ty);
#ifdef NOFRAMEWORK
		float_v front, back;
		for (int j = 0; j < SimdWidth && i + j < size; ++j) {
			front[j] = veloTracks[i + j].veloSegment ? veloTracks[i + j].veloSegment->front().z : 100.f;
			back[j] = veloTracks[i + j].veloSegment ? veloTracks[i + j].veloSegment->back().z : 200.f;
		}
		const float_mask_v isForward = front < back;
#else
		float_mask_v isForward;
		for (int j = 0; j < SimdWidth && i + j < size; ++j) {
			isForward[j] = !veloTracks[i + j]->checkFlag(LHCb::Event::v2::Track::Flag::Backward);
		}
#endif

		float_mask_v mask = withinAcceptance && isForward;
		int maski = mask.toInt();
		int popcount = PopCount(maski);

		const float_v stateX = SimdElement(veloEndStates.x, i);
		const float_v stateY = SimdElement(veloEndStates.y, i);
		const float_v stateZ = SimdElement(veloEndStates.z, i);
		const float_v stateTx = SimdElement(veloEndStates.tx, i);
		const float_v stateTy = SimdElement(veloEndStates.ty, i);
		const float_v stateQop = SimdElement(veloEndStates.qop, i);
		pext_prune256_epi32(stateX, maski).store(filteredStates.x.data() + writeIndex);
		pext_prune256_epi32(stateY, maski).store(filteredStates.y.data() + writeIndex);
		pext_prune256_epi32(stateZ, maski).store(filteredStates.z.data() + writeIndex);
		pext_prune256_epi32(stateTx, maski).store(filteredStates.tx.data() + writeIndex);
		pext_prune256_epi32(stateTy, maski).store(filteredStates.ty.data() + writeIndex);
		pext_prune256_epi32(stateQop, maski).store(filteredStates.qop.data() + writeIndex);

		for (int j = 0, writeIndexMod = writeIndex; j < SimdWidth && i + j < size; ++j) {
			if (mask[j]) {
				filteredTracks[writeIndexMod] = veloTracks[i + j];
				++writeIndexMod;
			}
		}

		writeIndex += popcount;
	}
	filteredTracks.resize(writeIndex);
	filteredStates.resize(writeIndex);

	return { std::move(filteredTracks), std::move(filteredStates) };
}


ExtrapolatedStatesS PrVeloUtOpt::ExtrapolateStates(const StatesS& states, float z, const Alloc& alloc) {
	size_t size = states.size();
	ExtrapolatedStatesS result{ alloc };
	result.resize(size);
	result.z = z;
	float_v zv = z;

	for (size_t i = 0; i < size; i += SimdWidth) {
		float_v zDiff = zv - SimdElement(states.z, i);
		float_v x = SimdElement(states.x, i) + zDiff * SimdElement(states.tx, i);
		float_v y = SimdElement(states.y, i) + zDiff * SimdElement(states.ty, i);
		SimdElement(result.x, i, x);
		SimdElement(result.y, i, y);
	}

	return result;
}


StateBinLocationsS PrVeloUtOpt::GetBinLocations(const UtHitSpacePartition& hits, const ExtrapolatedStatesS& layerStates, size_t& totalNumBins, const Alloc& alloc) {
	size_t numLayerStates = layerStates.size();
	size_t numLayerStatesSimd = numLayerStates & ~(SimdWidth - 1);
	StateBinLocationsS binLocations{ alloc };
	binLocations.resize(numLayerStates);

	totalNumBins = 0;
	uint16_v totalNumBinsVec = uint16_v::Zero(); // Overflows at 65535, should still be fine as we have less than 65535 hits for all the tracks, right?
	size_t i = 0;
	for (; i < numLayerStatesSimd; i += SimdWidth) {
		float_v x = SimdElement(layerStates.x, i);
		float_v y = SimdElement(layerStates.y, i);
		auto [index, count] = hits.GetDoubleBin(x, y, float_v(layerStates.z));

		SimdElement(binLocations.index, i, index);
		SimdElement(binLocations.count, i, count);
		totalNumBinsVec += count;
	}
	totalNumBins += totalNumBinsVec.sum();
	for (; i < numLayerStates; ++i) {
		float x = layerStates.x[i];
		float y = layerStates.y[i];
		auto [index, count] = hits.GetDoubleBin(x, y, layerStates.z);

		binLocations.index[i] = index;
		binLocations.count[i] = count;
		totalNumBins += count;
	}

	return binLocations;
}


stack_vector<float> PrVeloUtOpt::GetHitTolerances(const StatesS& states, const ExtrapolatedStatesS& layerStates, const Alloc& alloc) {
	const size_t numLayerStates = layerStates.size();

	stack_vector<float> result(numLayerStates, alloc);

	for (size_t i = 0; i < numLayerStates; i += SimdWidth) {
		const float_v trackTx = SimdElement(states.tx, i);
		const float_v trackTy = SimdElement(states.ty, i);

		const float_v adaptiveTolerance = GatherHitsTolerance(trackTx, trackTy);
		SimdElement(result, i, adaptiveTolerance);
	}

	return result;
}


std::pair<GatheredHitsS, stack_vector<uint32_t>> PrVeloUtOpt::GatherHits(const UtHitSpacePartition& hits, const StatesS& states, const ExtrapolatedStatesS& layerStates, const Alloc& alloc) {
	const UtHits& hitsContainer = hits.GetPartitionedHits();
	const size_t numLayerStates = layerStates.size();

	size_t sumBinSizes;
	StateBinLocationsS binLocations = GetBinLocations(hits, layerStates, sumBinSizes, alloc);
	stack_vector<float> tolerances = GetHitTolerances(states, layerStates, alloc);

	GatheredHitsS gatheredHits{ alloc };
	stack_vector<uint32_t> offsets{ alloc };
	gatheredHits.resize(sumBinSizes);
	offsets.resize(states.size());

	size_t writeIndex = 0;
	const float trackZ = layerStates.z;

	for (size_t i = 0; i < numLayerStates; ++i) {
		const auto index = binLocations.index[i];
		const auto count = binLocations.count[i];
		const float trackX = layerStates.x[i];
		const float trackY = layerStates.y[i];
		const float trackTx = states.tx[i];
		const float trackTy = states.ty[i];

		const float adaptiveTolerance = tolerances[i];

		for (int i = 0; i < count; i += SimdWidth) {
			auto totalIndex = index + i;
			uint32_v totalIndexSimd = uint32_v(totalIndex) + uint32_v::IndexesFromZero();

			float_v hitX0(hitsContainer.x0.data() + totalIndex);
			float_v hitZ0(hitsContainer.z0.data() + totalIndex);
			float_v hitY0(hitsContainer.y0.data() + totalIndex);
			float_v hitY1(hitsContainer.y1.data() + totalIndex);
			float_v hitDxdy(hitsContainer.dxdy.data() + totalIndex);
			float_v hitError(hitsContainer.error.data() + totalIndex);

			const float_v diffZ = hitZ0 - trackZ;
			const float_v trackAccurateX = trackX + diffZ * trackTx;
			const float_v trackAccurateY = trackY + diffZ * trackTy;
			const float_v hitAccurateX = hitX0 + hitDxdy * trackAccurateY;

			const float_mask_v mask = Vc::abs(hitAccurateX - trackAccurateX) < float_v(adaptiveTolerance)
									  && Vc::simd_cast<float_mask_v>(totalIndexSimd < uint32_v(index + count))
									  && hitY0 < trackY && trackY < hitY1;
			int maski = mask.toInt();
			int popcount = PopCount(maski);

			pext_prune256_epi32(hitAccurateX, maski).store(gatheredHits.x.data() + writeIndex);
			pext_prune256_epi32(hitZ0, maski).store(gatheredHits.z.data() + writeIndex);
			pext_prune256_epi32(reciprocal(hitError * hitError), maski).store(gatheredHits.weight.data() + writeIndex);
			pext_prune256_epi32(totalIndexSimd, maski).store(gatheredHits.index.data() + writeIndex);

			writeIndex += popcount;
		}

		offsets[i] = writeIndex;
	}

	gatheredHits.resize(writeIndex);

	return { std::move(gatheredHits), std::move(offsets) };
}


uint32_t PrVeloUtOpt::FindLinesForTrack(const std::array<GatheredHitsS, 4>& gatheredHits,
										const std::array<uint32_t, 4>& beginIndices,
										const std::array<uint32_t, 4>& endIndices,
										SearchLayers layerConfig,
										UtTrackCandidatesS& outCandidates,
										size_t writeIndex) {
	const size_t writeIndexStart = writeIndex;
	uint32_t minHitsFound = 3;
	uint32_t passedCount = 0;
	constexpr float maxDiff = 0.5f;

	for (size_t ibase1 = beginIndices[layerConfig.base1]; ibase1 < endIndices[layerConfig.base1]; ++ibase1) {
		const float xbase1 = gatheredHits[layerConfig.base1].x[ibase1];
		const float zbase1 = gatheredHits[layerConfig.base1].z[ibase1];

		for (size_t ibase2 = beginIndices[layerConfig.base2]; ibase2 < endIndices[layerConfig.base2]; ++ibase2) {
			const float xbase2 = gatheredHits[layerConfig.base2].x[ibase2];
			const float zbase2 = gatheredHits[layerConfig.base2].z[ibase2];

			const float dxdz = (xbase1 - xbase2) / (zbase1 - zbase2);
			const float x0 = xbase2 - dxdz * zbase2;

			auto [minDiffExtra1, idxExtra1] = ScanExtraHits(gatheredHits[layerConfig.extra1], beginIndices[layerConfig.extra1], endIndices[layerConfig.extra1], x0, dxdz, maxDiff);
			auto [minDiffExtra2, idxExtra2] = ScanExtraHits(gatheredHits[layerConfig.extra2], beginIndices[layerConfig.extra2], endIndices[layerConfig.extra2], x0, dxdz, maxDiff);

			uint32_t numHitsFound = 2 + (minDiffExtra1 < maxDiff) + (minDiffExtra2 < maxDiff);
			if (numHitsFound == 4 && minHitsFound == 3) {
				minHitsFound = 4;
				writeIndex = writeIndexStart;
				passedCount = 0;
			}
			if (numHitsFound >= minHitsFound) {
				outCandidates.hitAccurateX[layerConfig.base1][writeIndex] = xbase1;
				outCandidates.hitAccurateX[layerConfig.base2][writeIndex] = xbase2;
				outCandidates.hitAccurateX[layerConfig.extra1][writeIndex] = gatheredHits[layerConfig.extra1].x[idxExtra1];
				outCandidates.hitAccurateX[layerConfig.extra2][writeIndex] = gatheredHits[layerConfig.extra2].x[idxExtra2];

				outCandidates.hitZ[layerConfig.base1][writeIndex] = zbase1;
				outCandidates.hitZ[layerConfig.base2][writeIndex] = zbase2;
				outCandidates.hitZ[layerConfig.extra1][writeIndex] = gatheredHits[layerConfig.extra1].z[idxExtra1];
				outCandidates.hitZ[layerConfig.extra2][writeIndex] = gatheredHits[layerConfig.extra2].z[idxExtra2];

				outCandidates.hitWeight[layerConfig.base1][writeIndex] = gatheredHits[layerConfig.base1].weight[ibase1];
				outCandidates.hitWeight[layerConfig.base2][writeIndex] = gatheredHits[layerConfig.base2].weight[ibase2];
				outCandidates.hitWeight[layerConfig.extra1][writeIndex] = float(minDiffExtra1 < maxDiff) * gatheredHits[layerConfig.extra1].weight[idxExtra1];
				outCandidates.hitWeight[layerConfig.extra2][writeIndex] = float(minDiffExtra2 < maxDiff) * gatheredHits[layerConfig.extra2].weight[idxExtra2];

				outCandidates.hitIndex[layerConfig.base1][writeIndex] = gatheredHits[layerConfig.base1].index[ibase1];
				outCandidates.hitIndex[layerConfig.base2][writeIndex] = gatheredHits[layerConfig.base2].index[ibase2];
				outCandidates.hitIndex[layerConfig.extra1][writeIndex] = gatheredHits[layerConfig.extra1].index[idxExtra1];
				outCandidates.hitIndex[layerConfig.extra2][writeIndex] = gatheredHits[layerConfig.extra2].index[idxExtra2];
				++passedCount;
				++writeIndex;
			}
		}
	}

	return passedCount;
}


UtTrackCandidatesS PrVeloUtOpt::FindLines(const std::array<GatheredHitsS, 4>& gatheredHits, const std::array<stack_vector<uint32_t>, 4>& gatheredOffests, const StatesS& states, const Alloc& alloc) {
	size_t numTracks = gatheredOffests[0].size();

	UtTrackCandidatesS trackCandidates{ alloc };
	trackCandidates.resize(numTracks * 4);
	size_t writeIndex = 0;

	std::array<uint32_t, 4> hitIndices = { 0, 0, 0, 0 };

	for (size_t trackIdx = 0; trackIdx < numTracks; ++trackIdx) {
		std::array<uint32_t, 4> currentIndices;
		for (int i = 0; i < 4; ++i) {
			currentIndices[i] = gatheredOffests[i][trackIdx];
		}

		const auto& [combinatorics1, combinatorics2] = GetLineCombinatorics(hitIndices, currentIndices);
		const auto& [config1, config2] = GetSearchConfig(combinatorics1, combinatorics2);
		const auto& [minComb, maxComb] = std::minmax(combinatorics1, combinatorics2);

		if (trackCandidates.size() < writeIndex + maxComb) {
			trackCandidates.resize(writeIndex + maxComb);
		}

		uint32_t numCandidates = 0;
		if (maxComb > 0) {
			numCandidates = FindLinesForTrack(gatheredHits, hitIndices, currentIndices, config1, trackCandidates, writeIndex);
		}
		if (numCandidates == 0 && minComb > 0) {
			numCandidates = FindLinesForTrack(gatheredHits, hitIndices, currentIndices, config2, trackCandidates, writeIndex);
		}

		for (uint32_t i = 0; i < numCandidates; ++i) {
			trackCandidates.trackX[writeIndex + i] = states.x[trackIdx];
			trackCandidates.trackY[writeIndex + i] = states.y[trackIdx];
			trackCandidates.trackTx[writeIndex + i] = states.tx[trackIdx];
			trackCandidates.trackTy[writeIndex + i] = states.ty[trackIdx];
			trackCandidates.trackZ[writeIndex + i] = states.z[trackIdx];
			trackCandidates.trackIndex[writeIndex + i] = trackIdx;
		}

		writeIndex += numCandidates;
		hitIndices = currentIndices;
	}

	trackCandidates.resize(writeIndex);
	return trackCandidates;
}


LinearSystem PrVeloUtOpt::CreateFitLinearSystem(float_v zKink, float_v xTrackKink, float_v weightVelo, const UtTrackCandidatesS& candidates, uint32_t candidateIdx) {
	const float_v zDiff = zUnitScaling * (zKink - zMidUT);

	LinearSystem equationSystem = {
		std::array<float_v, 6>{ weightVelo, weightVelo * zDiff, weightVelo * zDiff * zDiff, 0.0f, 0.0f, 0.0f },
		std::array<float_v, 3>{ weightVelo * xTrackKink, weightVelo * xTrackKink * zDiff, 0.0f }
	};

	for (int layer = 0; layer < 4; ++layer) {
		const float_v ui = SimdElement(candidates.hitAccurateX[layer], candidateIdx);
		const float_v dz = zUnitScaling * (SimdElement(candidates.hitZ[layer], candidateIdx) - zMidUT);
		const float_v w = SimdElement(candidates.hitWeight[layer], candidateIdx);
		const float_v t = sineFiberAngles[layer]; // Sine of layer fiber angle.

		equationSystem.A[0] += w;
		equationSystem.A[1] += w * dz;
		equationSystem.A[2] += w * dz * dz;
		equationSystem.A[3] += w * t;
		equationSystem.A[4] += w * dz * t;
		equationSystem.A[5] += w * t * t;
		equationSystem.b[0] += w * ui;
		equationSystem.b[1] += w * ui * dz;
		equationSystem.b[2] += w * ui * t;
	}

	return equationSystem;
}


std::array<float_v, 2> PrVeloUtOpt::LinearRegressionFit(const UtTrackCandidatesS& candidates, uint32_t candidateIdx) {
	float_v sumx = float_v::Zero();
	float_v sumy = float_v::Zero();
	float_v sumw = float_v::Zero();

	const auto x = [&](int i) { return SimdElement(candidates.hitZ[i], candidateIdx); };
	const auto y = [&](int i) { return SimdElement(candidates.hitAccurateX[i], candidateIdx); };
	const auto w = [&](int i) { return SimdElement(candidates.hitWeight[i], candidateIdx); };

	for (int i = 0; i < 4; ++i) {
		sumx += x(i) * w(i);
		sumy += y(i) * w(i);
		sumw += w(i);
	}

	float_v rcpsumw = reciprocal(sumw);
	float_v meanx = sumx * rcpsumw;
	float_v meany = sumy * rcpsumw;

	float_v ssx = float_v::Zero();
	float_v sp = float_v::Zero();
	for (int i = 0; i < 4; ++i) {
		float_v diffx = x(i) - meanx;
		float_v diffy = y(i) - meany;
		ssx += diffx * diffx * w(i);
		sp += diffx * diffy * w(i);
	}

	// y = a + bx
	float_v b = sp / ssx;
	float_v a = meany - b * meanx;
	return { a, b };
}


inline float_v PrVeloUtOpt::EstimateQopInitial(float_v zKink, float_v xTrackKink, float_v x, float_v z, float_v tx, float_v ty, const UtTrackCandidatesS& candidates, uint32_t candidateIdx) {
	const float_v xTrackMidUt = x + tx * (zMidUT - z);


	auto [a, b] = LinearRegressionFit(candidates, candidateIdx);

	FittedTrackParameters fitParamsInitial;
	fitParamsInitial.xMidUt = a + b * zMidUT;
	fitParamsInitial.txMidUt = b;
	fitParamsInitial.yOffsetMidUt = 0.0f;

	float_v qop = CalculateQop(fitParamsInitial, zKink, xTrackKink, x, z, ty);
	return qop;
}


FittedTrackPropertiesS PrVeloUtOpt::FitTrackCandidates(const UtTrackCandidatesS& candidates, const Alloc& alloc) {
	size_t numCandidates = candidates.size();
	FittedTrackPropertiesS result{ alloc };
	result.resize(numCandidates);

	for (size_t candidateIdx = 0; candidateIdx < numCandidates; candidateIdx += SimdWidth) {
		const float_v x = SimdElement(candidates.trackX, candidateIdx);
		const float_v tx = SimdElement(candidates.trackTx, candidateIdx);
		const float_v ty = SimdElement(candidates.trackTy, candidateIdx);
		const float_v z = SimdElement(candidates.trackZ, candidateIdx);
		const float_v y = SimdElement(candidates.trackY, candidateIdx);

		const float_v qpxz2p = CalculateQpxz2p(x, z, tx, ty);

		const float_v zKink = EstimateKinkPosition(ty);
		const float_v xTrackKink = x + tx * (zKink - z);
		const float_v qopInitial = qpxz2p * EstimateQopInitial(zKink, xTrackKink, x, z, tx, ty, candidates, candidateIdx);
		const float_v weightVelo = CalculateWeightVelo(qopInitial, ty);

		LinearSystem fitSystem = CreateFitLinearSystem(zKink, xTrackKink, weightVelo, candidates, candidateIdx);

		CholeskyDecompInplace(fitSystem.A);
		CholeskySolveInplace(fitSystem.A, fitSystem.b);

		const FittedTrackParameters fitParams = {
			fitSystem.b[0],
			fitSystem.b[1] * zUnitScaling,
			fitSystem.b[2],
		};

		const float_v chi2 = CalculateChi2(xTrackKink, fitParams, zKink, weightVelo, candidates, candidateIdx);
		const float_v qop = qpxz2p * CalculateQop(fitParams, zKink, xTrackKink, x, z, ty);

		SimdElement(result.x, candidateIdx, fitParams.xMidUt);
		SimdElement(result.y, candidateIdx, fitParams.yOffsetMidUt + y + ty * (zMidUT - z));
		SimdElement(result.tx, candidateIdx, fitParams.txMidUt);
		SimdElement(result.chi2, candidateIdx, chi2);
		SimdElement(result.qop, candidateIdx, qop);
	}

	return result;
}


stack_vector<float> PrVeloUtOpt::ScoreCandidates(const UtTrackCandidatesS& candidates, const FittedTrackPropertiesS& fit, const Alloc& alloc) {
	size_t numCandidates = candidates.size();

	stack_vector<float> scores{ alloc };
	scores.resize(numCandidates);

	for (size_t i = 0; i < numCandidates; i += SimdWidth) {
		const float_v chi2 = SimdElement(fit.chi2, i);
		const float_v qop = SimdElement(fit.qop, i);
		const float_v tx = SimdElement(fit.tx, i);
		const float_v ty = SimdElement(candidates.trackTy, i);

		const float_mask_v potentialGhosts = MarkPotentialGhosts(chi2, qop, tx, ty);
		float_v score = -chi2 + maxChi2;
		score(potentialGhosts) = 0.0f;
		SimdElement(scores, i, score);
	}

	return scores;
}


stack_vector<uint32_t> PrVeloUtOpt::CandidateRunLengths(const UtTrackCandidatesS& candidates, const Alloc& alloc) {
	stack_vector<uint32_t> counts{ alloc };
	counts.resize(candidates.size());
	int32_t offset = -1;
	int32_t count = 0;
	uint32_t streakTrackIdx = 20000000; // Just anything that is not a valid value.
	int32_t writeIndex = 0;
	for (int i = 0; i < candidates.size(); ++i) {
		uint32_t trackIdx = candidates.trackIndex[i];
		int32_t same = int32_t(trackIdx == streakTrackIdx); // 1 if streak of extension-candidates for the same track continues, 0 if new track.
		writeIndex -= same;
		count *= same; // Set count to zero if new streak.
		count += 1; // Increase streak size.
		counts[writeIndex] = count;
		streakTrackIdx = trackIdx;
		++writeIndex;
	}
	const int32_t trackCount = writeIndex;
	counts.resize(trackCount);

	return counts;
}


stack_vector<uint32_t> PrVeloUtOpt::SelectBestCandidates(const UtTrackCandidatesS& candidates, const stack_vector<float>& scores, const Alloc& alloc) {
	stack_vector<uint32_t> candidatesPerTrackCounts = CandidateRunLengths(candidates, alloc);
	size_t trackCount = candidatesPerTrackCounts.size();

	stack_vector<uint32_t> bestCandidates{ alloc };
	bestCandidates.resize(trackCount);
	uint32_t writeIndex = 0;
	uint32_t offset = 0;
	for (size_t trackIdx = 0; trackIdx < trackCount; ++trackIdx) {
		uint32_t count = candidatesPerTrackCounts[trackIdx];

		float maxScore = 0.0f;
		int maxScoreIdx = offset;
		for (int j = offset; j < offset + count; ++j) {
			const float score = scores[j];
			bool better = score > maxScore;
			maxScore = better ? score : maxScore;
			maxScoreIdx = better ? j : maxScoreIdx;
		}

		int32_t success = int32_t(maxScore > 0.0f);
		bestCandidates[writeIndex] = success ? maxScoreIdx : bestCandidates[writeIndex];
		writeIndex += success;
		offset += count;
	}

	bestCandidates.resize(writeIndex);
	return bestCandidates;
}


AssembledTracks PrVeloUtOpt::AssembleTracks(const stack_vector<uint32_t>& selectedCandidates, const UtTrackCandidatesS& candidates, const FittedTrackPropertiesS& fitResults) {
	AssembledTracks tracks;
	tracks.resize(selectedCandidates.size());

	size_t writeIndex = 0;
	for (auto selectedCandidateIdx : selectedCandidates) {
		tracks.trackIndex[writeIndex] = candidates.trackIndex[selectedCandidateIdx];
		tracks.qop[writeIndex] = fitResults.qop[selectedCandidateIdx];

		for (size_t layer = 0; layer < 4; ++layer) {
			bool validIndex = candidates.hitWeight[layer][selectedCandidateIdx] > 0.0f;
			uint32_t sourceIndex = candidates.hitIndex[layer][selectedCandidateIdx];
			tracks.hitIndices[layer][writeIndex] = validIndex ? sourceIndex : uint32_t(-1);
		}

		++writeIndex;
	}

	return tracks;
}

} // namespace opt