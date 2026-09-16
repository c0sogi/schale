# 캐시 정책과 트래픽

SchaleDB 서버에 대한 반복 요청을 줄이기 위한 **로컬 애플리케이션 캐시**입니다. 프로그램이 실행되지 않을 때 백그라운드 폴링하지 않습니다.

## 읽기와 증분 갱신

1. 정규화한 URL로 인덱스를 찾고 현재 객체의 SHA-256을 검증합니다. 만료 전에는 HTTP 요청 0회입니다.
2. 자동 모드에서 만료된 항목만 GET합니다. 저장된 ETag/Last-Modified가 있으면 `If-None-Match`/`If-Modified-Since`를 보냅니다. HEAD로 한 번 확인하고 다시 GET하는 이중 요청은 하지 않습니다.
3. 서버가 304를 반환하면 본문 전송 없이 확인 시각과 만료 시각만 갱신합니다. 200이면 형식을 검증한 뒤 SHA-256 객체를 원자적으로 저장합니다. 같은 바이트는 URL이 달라도 하나의 객체를 공유합니다.
4. 이전 내용은 버전 객체로 보존합니다. SQLite에는 최초 관측, 다운로드, 서버 확인, 접근, 만료, 재시도 시각과 오류가 기록됩니다. `history`는 최신 이벤트부터 보여줍니다.

기본 TTL은 JSON **7일**, 이미지 **30일**입니다. TTL 안의 원격 변경은 감지하지 않습니다. 더 빠른 반영이 필요하면 TTL을 줄이거나 수동 재검증하세요. 조건부 요청은 [HTTP 표준](https://www.rfc-editor.org/rfc/rfc9110.html#section-13.1)에 따릅니다. 서버가 validator를 제공하지 않거나 무시하면 만료 시 해당 파일 전체를 받습니다.

증분은 **파일 단위**입니다. JSON의 일부 학생이 바뀌어도 해당 JSON 파일은 전부 받을 수 있습니다. 자체적인 원격 변경분 API나 전체 사이트 크롤링은 사용하지 않습니다.

## 모드와 수동 관리

```powershell
schale cache policy --mode auto
schale cache policy --json-ttl 604800 --image-ttl 2592000 --min-request-interval 0.5
schale cache update                  # 사용했던 항목 중 만료/누락만
schale cache update --refresh        # 만료 전도 조건부 확인
schale cache info
schale cache history --limit 30
schale cache verify                 # 네트워크 없이 현재 객체와 모델 해시 검사
```

- `auto`: 필요한 항목이 없으면 받고, 만료됐으면 재검증합니다.
- `manual`: 유효한 로컬 객체는 만료돼도 재사용합니다. 처음 필요한 누락 항목은 다운로드합니다. 명시적인 `cache update`로 만료 항목을 확인할 수 있습니다.
- `offline`: 만료와 관계없이 체크섬이 맞는 객체만 사용합니다. 누락·손상은 오류이며 네트워크 요청은 없습니다.

정책은 `cache-policy.json`에 저장됩니다. 일시적으로는 `$env:SCHALE_OFFLINE="1"`을 사용하면 `--refresh`보다도 우선하여 네트워크를 차단합니다. API는 `HttpCache.fetch(..., mode="offline")`처럼 호출별 정책도 지원합니다.

빈 캐시를 미리 준비할 때만 명시적 범위를 사용하세요.

```powershell
schale cache update --scope data --region kr  # 기준 JSON 6종
schale cache update --scope students         # 학생 정의 + 필요한 스킬 아이콘
schale cache update --scope equipment        # 장비 정의 + 아이콘
```

초상화를 일괄 다운로드하지 않습니다. 학생 식별에서 필요한 후보만 받습니다. 기본 `--scope used`는 사용한 리소스만 다루며 새로 추가된 전체 데이터를 탐색하지 않습니다. 학생 추출 시 갱신한 카탈로그에 새 스킬이 있으면 그 스킬 파일만 추가됩니다.

## 동시 실행과 실패

같은 캐시 루트의 프로세스들은 파일 잠금을 공유합니다. 잠금을 얻은 후 다시 캐시 상태를 확인하므로 같은 파일을 동시에 요청해도 첫 다운로드를 재사용합니다. HTTP 연결은 Session으로 재사용하고, 기본 요청 시작 간격은 최소 0.5초이며 동시에 하나만 전송합니다. 이 제한은 **같은 로컬 캐시 루트**에 적용됩니다. 서로 다른 PC나 별도 캐시 루트까지 분산 조정하지는 않습니다.

실패 시 기본 60초부터 지수적으로 재시도를 유예하고 404/410은 하루 동안 반복 요청하지 않습니다. 429/503의 Retry-After는 호스트 전체에 적용합니다. 유효한 이전 객체가 있으면 stale 상태로 반환하고 경고·오류를 기록하며, 손상된 객체를 성공값으로 반환하지 않습니다. `cache update`는 stale/실패가 남으면 비정상 종료 코드로 알려줍니다. 수동 refresh도 재시도 유예를 무시하지 않습니다.

형식 검사는 JSON 객체와 PNG/JPEG/WebP 기본 컨테이너·길이를 확인합니다. 이미지의 실제 디코딩은 비전 입력 단계에서 추가 확인합니다. 체크섬은 로컬 무결성 검증이며 서버 내용의 의미적 정답이나 배포자 서명이 아닙니다.

## 저장 구조와 기존 파일

```text
<SCHALE_CACHE_DIR 또는 사용자 홈/.schale/cache>/
├─ cache-policy.json
├─ http-v1/
│  ├─ index.sqlite3       # URL 상태·버전·시간순 이벤트
│  ├─ requests.lock      # 프로세스 간 중복/폭주 억제
│  └─ objects/<앞2자>/<SHA-256>
├─ vision-resources/     # 별도 로컬 참조 번들·내용별 버전·현재 선택
└─ student-numeric/      # 별도 설치하는 검증된 ONNX 번들
```

기존 flat JSON, assets/skills·portraits, 패키지 아이콘은 처음 사용할 때 복사·해시 등록하여 재사용하며 원본을 삭제하지 않습니다. 오래된 seed는 정책에 따라 재검증할 수 있습니다. 처음 등록한 해시는 기존 파일의 현재 내용을 기준으로 삼으며 과거 다운로드의 진위를 소급해 검증하지는 않습니다. 이후 손상은 캐시 hit에서도 감지합니다.

모델은 신뢰할 수 있는 로컬 ZIP/폴더로 설치하며 원격 최신 버전을 자동 검색하거나 교체하지 않습니다. SHA-256은 설치 및 doctor/verify 때 검사합니다. 모델을 배포하는 공식 manifest나 endpoint가 정해지기 전에는 임의 URL에서 가져오지 않습니다.

이전 객체·이력을 자동 삭제하지 않아 재사용과 추적성을 보존합니다. 디스크 용량 상한/LRU eviction/원격 모델 업데이트는 현재 제공하지 않습니다. 사용자 입력 영상과 계정 JSON은 이 캐시에 저장하지 않습니다.
