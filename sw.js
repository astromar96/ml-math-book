/* Service worker for the ML math book.
   Everything the book needs is same-origin and vendored, so the whole app
   works with no network at all once it has been opened once.
   Bump VERSION to ship a new build; the old cache is dropped on activate. */
'use strict';

var VERSION = 'v1';
var CACHE = 'mlmath-' + VERSION;

/* KaTeX ships one woff2 per font face; the book can reach any of them. */
var KATEX_FONTS = [
  'AMS-Regular',
  'Caligraphic-Bold',
  'Caligraphic-Regular',
  'Fraktur-Bold',
  'Fraktur-Regular',
  'Main-Bold',
  'Main-BoldItalic',
  'Main-Italic',
  'Main-Regular',
  'Math-BoldItalic',
  'Math-Italic',
  'SansSerif-Bold',
  'SansSerif-Italic',
  'SansSerif-Regular',
  'Script-Regular',
  'Size1-Regular',
  'Size2-Regular',
  'Size3-Regular',
  'Size4-Regular',
  'Typewriter-Regular',
].map(function (name) { return 'vendor/katex/fonts/KaTeX_' + name + '.woff2'; });

/* Google Fonts woff2, vendored. Hashed names come straight from Google's CSS. */
var WEB_FONTS = [
  'RrQKbpV-9Dd1b1OAGA6M9PkyDuVBeN2DHV2mLXgzTrc.woff2',
  'RrQKbpV-9Dd1b1OAGA6M9PkyDuVBeN2GHV2mLXgz.woff2',
  'RrQKbpV-9Dd1b1OAGA6M9PkyDuVBeN2IHV2mLXgzTrc.woff2',
  'SLXVc1nY6HkvangtZmpQdkhzfH5lkSscQyyS8p4_RHH1.woff2',
  'SLXVc1nY6HkvangtZmpQdkhzfH5lkSscRiyS8p4_RA.woff2',
  'SLXVc1nY6HkvangtZmpQdkhzfH5lkSscSCyS8p4_RHH1.woff2',
  'UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7W0Q5nw.woff2',
  'UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa25L7W0Q5n-wU.woff2',
  'tDbv2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKwBNntkaToggR7BYRbKPx7cwgknk-6nFg.woff2',
  'tDbv2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKwBNntkaToggR7BYRbKPxDcwgknk-4.woff2',
  'vEFH2_tTDB4M7-auWDN0ahZJW1ge6NmXq2ZbF8zBfb98SUr0aX2SzxEjst4.woff2',
  'vEFH2_tTDB4M7-auWDN0ahZJW1ge6NmXq2ZbF8zBfb98SUr6aX2SzxEj.woff2',
  'vEFI2_tTDB4M7-auWDN0ahZJW1gV8te1Xb7GlMo.woff2',
  'vEFI2_tTDB4M7-auWDN0ahZJW1gb8te1Xb7G.woff2',
].map(function (name) { return 'vendor/fonts/' + name; });

var PRECACHE = [
  './',
  'index.html',
  'english.html',
  'manifest.webmanifest',
  'book-ar.md',
  'book.md',
  'vendor/marked.min.js',
  'vendor/katex/katex.min.css',
  'vendor/katex/katex.min.js',
  'vendor/fonts-ar.css',
  'vendor/fonts-en.css',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
  'icons/apple-touch-icon.png'
].concat(KATEX_FONTS, WEB_FONTS);

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      /* Request each entry separately: one 404 should not void the whole install. */
      return Promise.all(PRECACHE.map(function (url) {
        return cache.add(new Request(url, { cache: 'reload' })).catch(function (err) {
          console.warn('[sw] could not precache', url, err);
        });
      }));
    })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (key) {
        if (key !== CACHE && key.indexOf('mlmath-') === 0) return caches.delete(key);
      }));
    }).then(function () {
      return self.clients.claim();
    })
  );
});

/* The page asks for this once the reader accepts a waiting update. */
self.addEventListener('message', function (event) {
  if (event.data === 'SKIP_WAITING') self.skipWaiting();
});

function revalidate(cache, request) {
  return fetch(request).then(function (response) {
    if (response && response.ok && response.type === 'basic') {
      return cache.put(request, response.clone()).then(function () { return response; });
    }
    return response;
  }).catch(function () { return null; });
}

/* Cache first, refresh in the background: the book opens instantly offline and
   still picks up a new build on the next visit. */
function respond(event, request) {
  return caches.open(CACHE).then(function (cache) {
    return cache.match(request, { ignoreSearch: true }).then(function (cached) {
      if (cached) {
        event.waitUntil(revalidate(cache, request));
        return cached;
      }
      return revalidate(cache, request).then(function (response) {
        if (response) return response;
        if (request.mode === 'navigate') {
          return cache.match('index.html').then(function (shell) {
            return shell || Response.error();
          });
        }
        return Response.error();
      });
    });
  });
}

self.addEventListener('fetch', function (event) {
  var request = event.request;
  if (request.method !== 'GET') return;
  if (new URL(request.url).origin !== self.location.origin) return;
  event.respondWith(respond(event, request));
});
