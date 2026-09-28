/*
 * HERoEHS lifelong 코어 자체 시험 — 합성 장면으로 삽입·격리·삭제 경로를 실제 rtabmap 으로 돌린다.
 *
 *   rtabmap-lifelong_selftest [작업 디렉토리]     (종료 코드 0 = 통과, 빌드는 같은 디렉토리 CMakeLists.txt)
 *
 * 장면: y 방향으로 옆걸음하는 로봇 앞 2 m 에 무작위 텍스처 벽 (depth 일정, ORB 특징 — GPU·torch 불필요).
 * 매핑 뒤 사용자 링크로 그래프를 눌러 최적화 pose ≠ raw odom pose (드리프트 보정량 C ≠ 0) 인 맵을 만든다 —
 * 삽입 링크의 프레임 혼용 버그(2026-09-28)가 드러나는 조건.
 *
 * 검사:
 *   T1 (WM)  삽입 = 격리(weight -2) · DB 에 격리로 기록 · 링크가 map 프레임 · 재최적화해도 제자리
 *            · 격리 중 전역 LC 후보·근접 결과에서 제외 · 격리 노드는 다음 삽입의 앵커가 아님
 *            · 해제 뒤 후보 복귀 · 삭제(메모리·이웃 링크·DB 행)
 *            · DB 재로드 뒤 위치추정 정상
 *   T2 (LTM) WM 상한으로 LTM 에 내려간 격리 노드의 해제·재격리(DB 직접) · 삭제(DB 행)
 *   T3       같은 세션 삽입 → LTM 이관 → 삭제 → 종료: WM 에 남은 앵커가 링크를 되살리지 않는다
 *   (격리 아닌 노드는 삭제 거부 — T1)
 */
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Statistics.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UStl.h>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <sqlite3.h>
#include <cstdio>
#include <cmath>
#include <map>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace rtabmap;

static int g_fail = 0;
#define CHECK(cond, ...) do { if(cond) { printf("  ok   "); printf(__VA_ARGS__); printf("\n"); } \
	else { printf("  FAIL "); printf(__VA_ARGS__); printf("   (%s:%d)\n", __FILE__, __LINE__); ++g_fail; } } while(0)

static const double kFx = 500.0, kCx = 320.0, kCy = 240.0, kWallX = 2.0;
static const double kTexRes = 0.004, kTexY0 = -3.0, kTexZ1 = 1.5;   // 벽 텍스처: y∈[-3,5], z∈[-1.5,1.5]

static cv::Mat makeTexture()
{
	cv::Mat noise(750, 2000, CV_8UC1);
	cv::RNG rng(1234);
	rng.fill(noise, cv::RNG::UNIFORM, 0, 255);
	cv::Mat tex;
	cv::GaussianBlur(noise, tex, cv::Size(0, 0), 2.0);
	cv::normalize(tex, tex, 0, 255, cv::NORM_MINMAX);
	return tex;
}

// 로봇 (x=0, y, yaw=0) 에서 본 영상 — 카메라는 base 원점, 광축 = base +x
static SensorData render(const cv::Mat & tex, double y, int id)
{
	cv::Mat rgb(480, 640, CV_8UC3), depth(480, 640, CV_16UC1, cv::Scalar(uint16_t(kWallX * 1000)));
	for(int v=0; v<480; ++v)
	{
		for(int u=0; u<640; ++u)
		{
			double Y = y - kWallX * (u - kCx) / kFx;     // optical x → base -y
			double Z = -kWallX * (v - kCy) / kFx;        // optical y → base -z
			int c = int((Y - kTexY0) / kTexRes), r = int((kTexZ1 - Z) / kTexRes);
			uchar g = (c >= 0 && c < tex.cols && r >= 0 && r < tex.rows) ? tex.at<uchar>(r, c) : 0;
			rgb.at<cv::Vec3b>(v, u) = cv::Vec3b(g, g, g);
		}
	}
	return SensorData(rgb, depth, CameraModel(kFx, kFx, kCx, kCy), id, double(id));
}

ParametersMap baseParams();
static ParametersMap locParams()
{
	ParametersMap p = baseParams();
	p[Parameters::kMemIncrementalMemory()] = "false";
	p[Parameters::kMemInitWMWithAllNodes()] = "true";
	p[Parameters::kRtabmapLoopThr()] = "0.02";   // 11 노드 합성 장면의 Bayes 사후확률은 0.03~0.08 — 기본 0.11 로는 안 걸린다
	return p;
}

ParametersMap baseParams()
{
	ParametersMap p;
	p[Parameters::kKpDetectorStrategy()] = "2";      // ORB
	p[Parameters::kVisFeatureType()] = "2";
	p[Parameters::kKpMaxFeatures()] = "400";
	p[Parameters::kVisMaxFeatures()] = "400";
	p[Parameters::kVisMinInliers()] = "10";
	p[Parameters::kMemRehearsalSimilarity()] = "1.0";   // 연속 노드 병합 금지
	p[Parameters::kRGBDLinearUpdate()] = "0";
	p[Parameters::kRGBDAngularUpdate()] = "0";
	p[Parameters::kRtabmapDetectionRate()] = "0";
	p[Parameters::kMemSTMSize()] = "2";
	// ASan 실행용: 계측 코어 + 비계측 /opt/ros libgtsam 은 Eigen 정렬 할당이 어긋나 GTSAM 안에서 bad-free 가 난다
	// (이 시험과 무관). LIFELONG_SELFTEST_OPTIMIZER=0 이면 외부 라이브러리 없는 TORO.
	if(getenv("LIFELONG_SELFTEST_OPTIMIZER"))
	{
		p[Parameters::kOptimizerStrategy()] = getenv("LIFELONG_SELFTEST_OPTIMIZER");
	}
	p[Parameters::kRGBDEnabled()] = "true";
	return p;
}

// DB 직접 조회 (rtabmap 이 연 상태에서도 읽기 가능)
static long long dbCount(const std::string & db, const std::string & sql)
{
	sqlite3 * h = 0;
	long long n = -1;
	if(sqlite3_open_v2(db.c_str(), &h, SQLITE_OPEN_READONLY, 0) == SQLITE_OK)
	{
		sqlite3_busy_timeout(h, 5000);
		sqlite3_stmt * st = 0;
		if(sqlite3_prepare_v2(h, sql.c_str(), -1, &st, 0) == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW)
		{
			n = sqlite3_column_int64(st, 0);
		}
		sqlite3_finalize(st);
	}
	sqlite3_close(h);
	return n;
}

static long long dbWeight(const std::string & db, int id, int timeoutMs = 0)
{
	char q[128];
	snprintf(q, sizeof(q), "SELECT CASE WHEN COUNT(*)=0 THEN -999 ELSE MAX(weight) END FROM Node WHERE id=%d", id);
	long long w = dbCount(db, q);
	for(int t=0; w == -999 && t < timeoutMs; t += 100)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));   // asyncSave 대기
		w = dbCount(db, q);
	}
	return w;
}

static long long rowsOf(const std::string & db, int id)
{
	char q[512];
	snprintf(q, sizeof(q),
		"SELECT (SELECT COUNT(*) FROM Node WHERE id=%d)+(SELECT COUNT(*) FROM Data WHERE id=%d)"
		"+(SELECT COUNT(*) FROM Feature WHERE node_id=%d)+(SELECT COUNT(*) FROM Link WHERE from_id=%d OR to_id=%d)",
		id, id, id, id, id);
	return dbCount(db, q);
}

static double dist(const Transform & a, const Transform & b) { return (a.inverse() * b).getNorm(); }

// 같은 자리(y)를 odom 을 조금씩 움직이며 frames 번 처리 — 받아들여진 LC·근접 대상 id 를 모은다.
// likelihood 에 excluded 가 한 번이라도 들어오면 likelihoodHit 을 켠다.
static std::vector<int> localize(Rtabmap & r, const cv::Mat & tex, double y, float odomY, int frames, int & idSeq,
		int watch = 0, bool * likelihoodHit = 0)
{
	std::vector<int> ids;
	for(int i=0; i<frames; ++i)
	{
		r.process(render(tex, y, idSeq++), Transform(0, odomY + 0.0005f * i, 0, 0, 0, 0), cv::Mat::eye(6, 6, CV_64FC1));
		const Statistics & st = r.getStatistics();
		if(st.loopClosureId() > 0) ids.push_back(st.loopClosureId());
		if(st.proximityDetectionId() > 0) ids.push_back(st.proximityDetectionId());
		if(likelihoodHit && watch && st.likelihood().count(watch)) *likelihoodHit = true;
		if(getenv("LIFELONG_SELFTEST_DEBUG"))
		{
			const std::map<std::string, float> & d = st.data();
			printf("    [dbg] y=%.3f hyp=%d val=%.3f accepted=%d rejected=%d inliers=%d prox=%d\n", y,
					(int)uValue(d, std::string("Loop/Highest_hypothesis_id/"), 0.0f), uValue(d, std::string("Loop/Highest_hypothesis_value/"), 0.0f),
					st.loopClosureId(), (int)uValue(d, std::string("Loop/Rejected_hypothesis/"), 0.0f),
					(int)uValue(d, std::string("Loop/Visual_inliers/"), 0.0f), st.proximityDetectionId());
		}
	}
	return ids;
}
static bool contains(const std::vector<int> & v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

int main(int argc, char ** argv)
{
	setvbuf(stdout, 0, _IONBF, 0);   // 중단돼도 진행 로그가 남게
	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kWarning);
	const std::string dir = argc > 1 ? argv[1] : ".";
	const std::string db = dir + "/lifelong_selftest.db";
	UFile::erase(db);
	const cv::Mat tex = makeTexture();
	const cv::Mat cov = cv::Mat::eye(6, 6, CV_64FC1);   // odom 링크 분산 1 — 약하게 (사용자 링크가 그래프를 누르도록)

	// ---------- 매핑: 11 노드, y = 0 .. 2.0 ----------
	printf("[map] 11 nodes + user link (graph compression → C != 0)\n");
	{
		Rtabmap r;
		ParametersMap p = baseParams();
		p[Parameters::kMemIncrementalMemory()] = "true";
		p[Parameters::kRGBDProximityBySpace()] = "false";
		p[Parameters::kRtabmapLoopThr()] = "0.99";
		r.init(p, db);
		for(int i=0; i<=10; ++i)
		{
			r.process(render(tex, 0.2 * i, i + 1), Transform(0, 0.2f * i, 0, 0, 0, 0), cov);
		}
		const int first = 1, last = r.getLastLocationId();
		cv::Mat info = cv::Mat::eye(6, 6, CV_64FC1) * 1e4;
		CHECK(r.addLink(Link(first, last, Link::kUserClosure, Transform(0, 1.5f, 0, 0, 0, 0), info)),
				"user link %d->%d added", first, last);
		r.close(true);
	}

	std::map<int, Transform> raw, opt;
	int anchorProbe = 0;
	double maxC = 0.0;
	// ---------- T1: WM 경로 ----------
	printf("[T1] localization, WM path\n");
	int N = 0;
	Transform mapPoseN;
	{
		Rtabmap r;
		r.init(locParams(), db);
		std::multimap<int, Link> links;
		r.getGraph(opt, links, true, true);
		r.getGraph(raw, links, false, true);
		for(std::map<int, Transform>::iterator iter=opt.begin(); iter!=opt.end(); ++iter)
		{
			if(raw.count(iter->first) && dist(raw.at(iter->first), iter->second) > maxC)
			{
				maxC = dist(raw.at(iter->first), iter->second);
				anchorProbe = iter->first;
			}
		}
		CHECK(maxC > 0.05, "map has drift correction C (max %.3f m at node %d) — frame-mix bug observable", maxC, anchorProbe);

		// 위치추정: 노드 6 (y=1.0) 과 같은 영상, odom 원점 근처
		int seq = 100;
		std::vector<int> loc = localize(r, tex, 1.0, 0.0f, 4, seq);
		CHECK(!loc.empty(), "localized (first accepted target %d)", loc.empty() ? 0 : loc.front());
		const Transform mapCorr = r.getMapCorrection() * Transform(0, 0.0015f, 0, 0, 0, 0);  // 마지막 프레임 odom 과 맞춤

		// 삽입: y=1.1 영상
		const Transform odomN(0, 0.1f, 0, 0, 0, 0);
		mapPoseN = mapCorr * odomN;
		cv::Mat lcov = cv::Mat::eye(6, 6, CV_64FC1) * 0.01;
		lcov.at<double>(3,3) = lcov.at<double>(4,4) = lcov.at<double>(5,5) = 0.005;
		N = r.ingestNode(render(tex, 1.1, seq++), mapPoseN, lcov);
		CHECK(N > 0, "ingested node %d", N);
		const Signature * sN = r.getMemory()->getSignature(N);
		CHECK(sN && sN->getWeight() == Memory::kQuarantinedWeight, "born quarantined (weight %d)", sN ? sN->getWeight() : 999);

		// 링크는 map 프레임: 앵커 최적화 pose × 링크 = 삽입 map pose
		std::map<int, Transform> optNow;
		r.getGraph(optNow, links, true, true);
		int nLinks = 0;
		double linkErr = 0.0;
		if(sN)
		{
			for(std::multimap<int, Link>::const_iterator iter=sN->getLinks().begin(); iter!=sN->getLinks().end(); ++iter)
			{
				if(iter->first == N || !optNow.count(iter->first)) continue;
				++nLinks;
				// 새 노드 쪽 링크 = N→anchor 변환 → anchor 기준 N = inverse
				Transform predicted = optNow.at(iter->first) * iter->second.transform().inverse();
				linkErr = std::max(linkErr, dist(predicted, mapPoseN));
			}
		}
		CHECK(nLinks == 2, "2 neighbor links (%d)", nLinks);
		CHECK(linkErr < 1e-3, "links are map-frame consistent (max err %.5f m)", linkErr);
		CHECK(optNow.count(N) && dist(optNow.at(N), mapPoseN) < 1e-3,
				"re-optimized graph keeps node at its map pose (%.5f m)", optNow.count(N) ? dist(optNow.at(N), mapPoseN) : -1.0);

		// 격리: N 과 같은 영상으로 여러 번 — 위치추정은 되되 N 은 후보(likelihood)·수락 대상 어디에도 없어야 한다
		bool seen = false;
		std::vector<int> quar = localize(r, tex, 1.1, 0.1f, 6, seq, N, &seen);
		CHECK(dbWeight(db, N, 5000) == Memory::kQuarantinedWeight, "quarantine persisted in DB after next cycle (weight %lld)", dbWeight(db, N));
		CHECK(!seen, "quarantined node never in loop-closure candidates");
		CHECK(!quar.empty() && !contains(quar, N), "localization continues on original nodes, never on the quarantined node (%zu accepted)", quar.size());

		// 격리 노드는 앵커가 아니다: N 과 같은 map pose 로 하나 더 삽입해도 N 에 붙지 않는다
		const int N1 = r.ingestNode(render(tex, 1.1, seq++), mapPoseN, lcov);
		bool toQuar = false;
		if(N1 > 0)
		{
			toQuar = r.getMemory()->getSignature(N1)->hasLink(N) || r.getMemory()->getSignature(N)->hasLink(N1);
		}
		CHECK(N1 > 0 && !toQuar, "2nd ingest at the quarantined node's pose does not anchor to it (node %d)", N1);
		CHECK(N1 > 0 && r.deleteNodes(std::vector<int>(1, N1)).size() == 1, "2nd ingested node deleted");

		// 해제 → 후보 복귀
		CHECK(r.setNodesQuarantined(std::vector<int>(1, N), false) == 1, "release applied");
		CHECK(r.getMemory()->getSignature(N)->getWeight() == 0 && dbWeight(db, N) == 0, "released in memory and DB");
		seen = false;
		std::vector<int> rel = localize(r, tex, 1.1, 0.1f, 16, seq, N, &seen);
		CHECK(seen, "released node is a candidate again");
		CHECK(contains(rel, N), "released node is actually used for localization (%zu accepted)", rel.size());
		CHECK(r.deleteNodes(std::vector<int>(1, N)).empty() && r.getMemory()->getSignature(N) != 0,
				"released (normal) node cannot be deleted");
		CHECK(r.deleteNodes(std::vector<int>(1, 1)).empty() && r.getMemory()->getSignature(1) != 0,
				"original map node cannot be deleted");

		// 재격리 → 삭제
		CHECK(r.setNodesQuarantined(std::vector<int>(1, N), true) == 1 && r.getMemory()->getSignature(N)->getWeight() == Memory::kQuarantinedWeight,
				"re-quarantine applied");
		std::vector<int> anchors;
		for(std::multimap<int, Link>::const_iterator iter=r.getMemory()->getSignature(N)->getLinks().begin();
			iter!=r.getMemory()->getSignature(N)->getLinks().end(); ++iter)
		{
			if(iter->first != N) anchors.push_back(iter->first);
		}
		CHECK(r.labelLocation(N, "lst_quarantined"), "labelled the quarantined node");
		std::vector<int> del = r.deleteNodes(std::vector<int>(1, N));
		CHECK(del.size() == 1 && del[0] == N, "deleteNodes returned %zu", del.size());
		CHECK(r.getMemory()->getSignature(N) == 0, "gone from memory");
		CHECK(r.getMemory()->getAllLabels().count(N) == 0, "its label is gone too");
		bool dangling = false;
		for(size_t i=0; i<anchors.size(); ++i)
		{
			const Signature * a = r.getMemory()->getSignature(anchors[i]);
			if(a && a->hasLink(N)) dangling = true;
		}
		CHECK(!dangling, "neighbors no longer link to it");
		std::map<int, Transform> optAfter;
		r.getGraph(optAfter, links, true, true);
		CHECK(optAfter.count(N) == 0, "not in optimized graph");
		CHECK(rowsOf(db, N) == 0, "DB rows gone (Node/Data/Feature/Link: %lld)", rowsOf(db, N));
		CHECK(r.deleteNodes(std::vector<int>(1, N)).empty(), "second delete is a no-op");
		std::vector<int> after = localize(r, tex, 1.0, 0.0f, 6, seq);
		CHECK(!after.empty() && !contains(after, N), "still localizes after deletion (%zu accepted)", after.size());
		r.close(true);
		CHECK(rowsOf(db, N) == 0, "still no rows after close (no re-save)");
		char q[160];
		snprintf(q, sizeof(q), "SELECT COUNT(*) FROM Link WHERE to_id NOT IN (SELECT id FROM Node) OR from_id NOT IN (SELECT id FROM Node)");
		CHECK(dbCount(db, q) == 0, "no dangling link rows in DB");
	}

	// ---------- 재로드 ----------
	printf("[reload] reopen DB and localize\n");
	{
		Rtabmap r;
		r.init(locParams(), db);
		int seq = 200;
		std::vector<int> loc = localize(r, tex, 0.6, 0.0f, 6, seq);
		CHECK(!loc.empty(), "localized after reload (%zu accepted)", loc.size());
		r.close(false);
	}

	// ---------- T2: LTM 경로 ----------
	printf("[T2] LTM path: ingest in session A, act on it from session B where it is not in WM\n");
	int N2 = 0;
	{
		Rtabmap r;
		r.init(locParams(), db);
		int seq = 300;
		std::vector<int> loc = localize(r, tex, 0.2, 0.0f, 4, seq);
		CHECK(!loc.empty(), "session A localized");
		cv::Mat lcov = cv::Mat::eye(6, 6, CV_64FC1) * 0.01;
		N2 = r.ingestNode(render(tex, 0.3, seq++), r.getMapCorrection() * Transform(0, 0.1015f, 0, 0, 0, 0), lcov);
		CHECK(N2 > 0, "ingested node %d", N2);
		localize(r, tex, 0.4, 0.3f, 2, seq);
		CHECK(dbWeight(db, N2, 5000) == Memory::kQuarantinedWeight, "quarantine persisted in DB");
		r.close(true);
	}
	{
		Rtabmap r;
		ParametersMap p = locParams();
		// LTM 경로 강제: WM 상한 + 최근 WM 면제 끔 → 격리 노드(weight -2, 가장 낮음)가 먼저 LTM 으로.
		// (InitWMWithAllNodes=false 는 로컬라이제이션 모드에서 Statistics 가 안 쌓여 삽입 노드를 늘 WM 에 올린다)
		p[Parameters::kRtabmapMemoryThr()] = "5";
		p[Parameters::kMemRecentWmRatio()] = "0";
		// 전역 면제 = 최고 가설 주변 (Bayes 예측 벡터 길이) 홉 — 기본 ~19 홉이면 12 노드 맵은 통째로 면제된다.
		// 실제 맵(수백 노드)에선 먼 노드가 이관되지만 이 작은 장면에선 3 홉으로 줄여야 LTM 경로가 열린다.
		p[Parameters::kBayesPredictionLC()] = "0.1 0.36 0.30";
		r.init(p, db);
		int seq = 360;
		for(int i=0; i<30 && r.getMemory()->getSignature(N2); ++i)
		{
			localize(r, tex, 1.6 + 0.01 * (i % 20), 1.5f + 0.01f * i, 1, seq);   // N2(y≈0.3)에서 먼 곳
			if(getenv("LIFELONG_SELFTEST_DEBUG"))
			{
				const std::map<std::string, float> & d = r.getStatistics().data();
				printf("    [dbg] WM=%zu N2 %s removed=%.0f retrieved=%.0f immunized=%.0f/%.0f\n", r.getMemory()->getWorkingMem().size(),
						r.getMemory()->getSignature(N2)?"in WM":"out",
						uValue(d, std::string("Memory/Signatures_removed/"), -1.0f), uValue(d, std::string("Memory/Signatures_retrieved/"), -1.0f),
						uValue(d, std::string("Memory/Immunized_locally/"), -1.0f), uValue(d, std::string("Memory/Immunized_globally/"), -1.0f));
			}
		}
		const bool inLTM = r.getMemory()->getSignature(N2) == 0;
		printf("  info N2 %s\n", inLTM ? "is in LTM" : "is in WM");
		CHECK(inLTM, "node from previous session is in LTM (LTM path exercised)");
		CHECK(r.setNodesQuarantined(std::vector<int>(1, N2), false) == 1 && dbWeight(db, N2) == 0, "release on LTM node written to DB");
		CHECK(r.setNodesQuarantined(std::vector<int>(1, N2), true) == 1 && dbWeight(db, N2) == Memory::kQuarantinedWeight,
				"re-quarantine on LTM node written to DB");
		CHECK(r.setNodesQuarantined(std::vector<int>(1, 999999), false) == 0, "unknown id reports 0");
		std::vector<int> del = r.deleteNodes(std::vector<int>(1, N2));
		CHECK(del.size() == 1 && rowsOf(db, N2) == 0, "LTM node deleted from DB (%lld rows)", rowsOf(db, N2));
		CHECK(r.deleteNodes(std::vector<int>(1, 999999)).empty(), "unknown id delete is a no-op");
		r.close(true);
		CHECK(rowsOf(db, N2) == 0, "still no rows after close");
		char q[160];
		snprintf(q, sizeof(q), "SELECT COUNT(*) FROM Link WHERE to_id NOT IN (SELECT id FROM Node) OR from_id NOT IN (SELECT id FROM Node)");
		CHECK(dbCount(db, q) == 0, "no dangling link rows in DB");
	}
	// ---------- T3: 같은 세션 LTM 삭제 — WM 에 남은 앵커가 쥔 링크가 저장 때 되살아나면 안 된다 ----------
	printf("[T3] same session: ingest -> WM->LTM -> delete -> close (anchors in WM must drop the link)\n");
	{
		Rtabmap r;
		ParametersMap p = locParams();
		p[Parameters::kRtabmapMemoryThr()] = "5";
		p[Parameters::kMemRecentWmRatio()] = "0";
		p[Parameters::kBayesPredictionLC()] = "0.1 0.36 0.30";
		r.init(p, db);
		int seq = 500;
		std::vector<int> loc = localize(r, tex, 0.2, 0.0f, 4, seq);
		CHECK(!loc.empty(), "localized");
		cv::Mat lcov = cv::Mat::eye(6, 6, CV_64FC1) * 0.01;
		const int N3 = r.ingestNode(render(tex, 0.3, seq++), r.getMapCorrection() * Transform(0, 0.1015f, 0, 0, 0, 0), lcov);
		CHECK(N3 > 0, "ingested node %d", N3);
		if(getenv("LIFELONG_SELFTEST_DEBUG") && N3 > 0)
		{
			printf("    [dbg] after ingest WM:");
			for(std::map<int, double>::const_iterator iter=r.getMemory()->getWorkingMem().begin(); iter!=r.getMemory()->getWorkingMem().end(); ++iter) printf(" %d", iter->first);
			printf(" | N3 links:");
			for(std::multimap<int, Link>::const_iterator iter=r.getMemory()->getSignature(N3)->getLinks().begin(); iter!=r.getMemory()->getSignature(N3)->getLinks().end(); ++iter) printf(" %d", iter->first);
			printf("\n");
		}
		// 앵커(2·3) 근처에서 몇 프레임 더 — 근접 노드 나이 갱신(updateAge)으로 앵커가 WM 에 늦게까지 남는다.
		// 실제 맵(WM 수백 노드)에선 격리 노드(weight -2)가 먼저 이관되고 앵커는 WM 에 남는 게 보통이다.
		localize(r, tex, 0.35, 0.15f, 3, seq);
		for(int i=0; i<30 && N3 > 0 && r.getMemory()->getSignature(N3); ++i)
		{
			localize(r, tex, 1.6 + 0.01 * (i % 20), 1.5f + 0.01f * i, 1, seq);
		}
		CHECK(N3 > 0 && r.getMemory()->getSignature(N3) == 0, "moved to LTM in the same session");
		std::vector<int> holders;   // WM 에서 N3 로 가는 링크를 쥔 노드 — 있어야 이 시험이 의미 있다
		for(std::map<int, double>::const_iterator iter=r.getMemory()->getWorkingMem().begin(); iter!=r.getMemory()->getWorkingMem().end(); ++iter)
		{
			const Signature * s = r.getMemory()->getSignature(iter->first);
			if(s && s->hasLink(N3)) holders.push_back(iter->first);
		}
		if(getenv("LIFELONG_SELFTEST_DEBUG"))
		{
			printf("    [dbg] WM:");
			for(std::map<int, double>::const_iterator iter=r.getMemory()->getWorkingMem().begin(); iter!=r.getMemory()->getWorkingMem().end(); ++iter) printf(" %d", iter->first);
			char qq[200];
			snprintf(qq, sizeof(qq), "SELECT COUNT(*) FROM Link WHERE from_id=%d", N3);
			printf("  | N3 link rows in DB: %lld\n", dbCount(db, qq));
		}
		CHECK(!holders.empty(), "anchors still in WM hold a link to it (%zu)", holders.size());
		std::vector<int> del = r.deleteNodes(std::vector<int>(1, N3));
		CHECK(del.size() == 1, "deleted while in LTM");
		bool back = false;
		for(size_t i=0; i<holders.size(); ++i)
		{
			const Signature * s = r.getMemory()->getSignature(holders[i]);
			if(s && s->hasLink(N3)) back = true;
		}
		CHECK(!back, "no in-memory back-link left");
		r.close(true);
		char q[160];
		snprintf(q, sizeof(q), "SELECT COUNT(*) FROM Link WHERE to_id NOT IN (SELECT id FROM Node) OR from_id NOT IN (SELECT id FROM Node)");
		CHECK(dbCount(db, q) == 0 && rowsOf(db, N3) == 0, "no dangling link rows after close (%lld)", dbCount(db, q));
		snprintf(q, sizeof(q), "SELECT COUNT(*) FROM Feature WHERE word_id>0 AND word_id NOT IN (SELECT id FROM Word)");
		CHECK(dbCount(db, q) == 0, "every stored feature's word is in the Word table (%lld missing)", dbCount(db, q));
	}
	{
		Rtabmap r;
		r.init(locParams(), db);
		int seq = 400;
		std::vector<int> loc = localize(r, tex, 1.4, 0.0f, 6, seq);
		CHECK(!loc.empty(), "final reload localizes (%zu accepted)", loc.size());
		r.close(false);
	}

	printf(g_fail ? "\nFAILED: %d check(s)\n" : "\nALL PASSED\n", g_fail);
	return g_fail ? 1 : 0;
}
