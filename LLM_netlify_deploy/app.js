/* LLM Foundations — App Logic v3 */

document.addEventListener('DOMContentLoaded', () => {
    loadDynamicContent();
    setupReadingProgress();
    setupScrollReveal();
    setupActiveNav();
    setupSidebarToggle();
    setupCollapsibles();
});

// ── Load advanced chapters ──────────────────────────
async function loadDynamicContent() {
    const container = document.getElementById('dynamic-content');
    if (!container) return;
    try {
        const resp = await fetch('content2.html');
        if (resp.ok) {
            container.innerHTML = await resp.text();
            setupScrollReveal();
            setupCollapsibles();
        }
    } catch (e) {
        container.innerHTML = `<div class="container" style="padding:60px 0;text-align:center">
            <div class="card"><h3>📡 Load Advanced Chapters</h3>
            <p>Serve via local server to load advanced chapters (Ch06–Ch14 + Interview Prep):</p>
            <div class="code-block"><code>python3 -m http.server 8080
# Then open: http://localhost:8080/LLM_Foundations/</code></div></div></div>`;
    }
}

// ── Reading Progress Bar ────────────────────────────
function setupReadingProgress() {
    const bar = document.getElementById('readingProgress');
    if (!bar) return;
    window.addEventListener('scroll', () => {
        const total = document.documentElement.scrollHeight - window.innerHeight;
        bar.style.width = total > 0 ? (window.scrollY / total * 100) + '%' : '0%';
    }, { passive: true });
}

// ── Scroll Reveal ───────────────────────────────────
function setupScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
    document.querySelectorAll('.reveal:not(.visible)').forEach(el => observer.observe(el));
}

// ── Active Sidebar Link ─────────────────────────────
function setupActiveNav() {
    const links = document.querySelectorAll('.sidebar-link');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                links.forEach(l => {
                    l.classList.toggle('active', l.getAttribute('href') === '#' + id);
                });
            }
        });
    }, { rootMargin: '-40% 0px -55% 0px' });
    document.querySelectorAll('section[id], div[id]').forEach(s => observer.observe(s));
}

// ── Sidebar Toggle (mobile) ─────────────────────────
function setupSidebarToggle() {
    const btn = document.getElementById('navToggle');
    const sb = document.getElementById('sidebar');
    if (!btn || !sb) return;
    btn.addEventListener('click', () => sb.classList.toggle('open'));
    document.addEventListener('click', (e) => {
        if (!sb.contains(e.target) && !btn.contains(e.target)) sb.classList.remove('open');
    });
}

// ── Collapsibles ────────────────────────────────────
function setupCollapsibles() {
    document.querySelectorAll('.collapsible-header').forEach(header => {
        if (header.dataset.bound) return;
        header.dataset.bound = '1';
        header.addEventListener('click', () => {
            header.closest('.collapsible').classList.toggle('open');
        });
    });
}
