/* ============================================================
   RAG: ZERO TO HERO — Application Logic v2.0
   ============================================================ */

'use strict';

document.addEventListener('DOMContentLoaded', () => {
    loadAdvancedContent();
    initProgressBar();
    initScrollReveal();
    initSidebar();
    initMobileNav();
    initQuiz();
});

/* ─── 1. Load Advanced Content ─────────────────────────────── */
async function loadAdvancedContent() {
    const container = document.getElementById('advanced-content');
    if (!container) return;

    try {
        const res = await fetch('./advanced.html');
        if (!res.ok) throw new Error('fetch failed');
        container.innerHTML = await res.text();

        // Re-init after content injection
        initScrollReveal();
        initSidebar();
        initQuiz();
    } catch (e) {
        container.innerHTML = `
      <section class="section">
        <div class="container">
          <div class="card">
            <div class="card-icon" style="background:rgba(245,158,11,.12);color:var(--orange)">🖥️</div>
            <h3>View Advanced Chapters</h3>
            <p>To view Chapters 6–16 (Agentic RAG, GraphRAG, RAPTOR, RAGAS, Production…),
            please serve this folder using a local HTTP server:</p>
            <div class="code-block">
              <div class="code-header">
                <div class="code-dots"><span></span><span></span><span></span></div>
                <span class="code-lang">Terminal</span>
              </div>
              <div class="code-body"><code>cd /Users/muhammadsharjeel/Learning_RAG/RAG_Zero_To_Hero
python3 -m http.server 8080
# Then open: http://localhost:8080</code></div>
            </div>
            <p style="margin-top:14px;color:var(--text-2)">This is required because browsers block loading local files due to CORS security policy. The content is inside <code>advanced.html</code> in this same folder.</p>
          </div>
        </div>
      </section>`;
    }
}

/* ─── 2. Reading Progress Bar ───────────────────────────────── */
function initProgressBar() {
    const bar = document.getElementById('progressBar');
    if (!bar) return;

    window.addEventListener('scroll', () => {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        bar.style.width = pct.toFixed(1) + '%';
    }, { passive: true });
}

/* ─── 3. Scroll Reveal Animations ───────────────────────────── */
function initScrollReveal() {
    const targets = document.querySelectorAll('.reveal:not(.observed)');
    if (!targets.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }, i * 60);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    targets.forEach(el => {
        el.classList.add('observed');
        observer.observe(el);
    });
}

/* ─── 4. Sidebar Active Link on Scroll ──────────────────────── */
function initSidebar() {
    const links = document.querySelectorAll('.sidebar-link');
    if (!links.length) return;

    const sectionIds = Array.from(links)
        .map(l => l.getAttribute('href')?.replace('#', ''))
        .filter(Boolean);

    const setActive = () => {
        const scrollY = window.scrollY + 120;
        let current = sectionIds[0];

        sectionIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.offsetTop <= scrollY) current = id;
        });

        links.forEach(link => {
            const target = link.getAttribute('href')?.replace('#', '');
            link.classList.toggle('active', target === current);
        });
    };

    window.addEventListener('scroll', setActive, { passive: true });
    setActive();
}

/* ─── 5. Mobile Nav Toggle ───────────────────────────────────── */
function initMobileNav() {
    const toggle = document.getElementById('navToggle');
    const sidebar = document.getElementById('sidebar');
    if (!toggle || !sidebar) return;

    toggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        toggle.textContent = sidebar.classList.contains('open') ? '✕' : '☰';
    });

    // Close on link click (mobile UX)
    sidebar.querySelectorAll('.sidebar-link').forEach(link => {
        link.addEventListener('click', () => {
            sidebar.classList.remove('open');
            toggle.textContent = '☰';
        });
    });

    // Close on outside tap
    document.addEventListener('click', (e) => {
        if (!sidebar.contains(e.target) && !toggle.contains(e.target)) {
            sidebar.classList.remove('open');
            toggle.textContent = '☰';
        }
    });
}

/* ─── 6. Interactive Quiz ────────────────────────────────────── */
function initQuiz() {
    document.querySelectorAll('.quiz-opt:not(.quiz-bound)').forEach(btn => {
        btn.classList.add('quiz-bound');
        btn.addEventListener('click', handleQuizAnswer);
    });
}

function handleQuizAnswer(e) {
    const btn = e.currentTarget;
    const qId = btn.dataset.q;
    const isCorrect = btn.dataset.correct === 'true';
    const card = document.getElementById(qId);
    if (!card || card.dataset.answered) return; // one attempt only

    card.dataset.answered = 'true';

    // Style all options
    card.querySelectorAll('.quiz-opt').forEach(opt => {
        opt.disabled = true;
        if (opt.dataset.correct === 'true') opt.classList.add('correct');
        else if (opt === btn && !isCorrect) opt.classList.add('wrong');
    });

    // Show feedback
    const fb = document.getElementById(qId + '-fb');
    if (!fb) return;
    fb.classList.add('show');

    const answers = {
        q1: {
            ok: '✅ Perfect! RAG = dynamic fact injection at inference. Fine-Tuning = baking behavior into weights at training. They complement each other — many production systems use both.',
            bad: '❌ Not quite. Fine-Tuning bakes knowledge into model weights at training time, but it\'s expensive, slow to update, and doesn\'t reliably prevent hallucination. RAG retrieves facts dynamically at inference time — much more flexible for factual knowledge.'
        },
        q2: {
            ok: '✅ Correct! These three failure modes (query quality, multi-hop reasoning, exact keyword matching) drive the entire Advanced RAG chapter. Each has a specific fix: HyDE/Multi-Query, Chain-of-Thought retrieval, and Hybrid Search respectively.',
            bad: '❌ Those are concerns, but not the core failure modes. The real problems are about RETRIEVAL QUALITY — the LLM itself is less often the bottleneck than getting the right chunks to it in the first place.'
        },
        q3: {
            ok: '✅ Exactly right! HyDE exploits the geometric structure of embedding space: questions and answers are in different "neighborhoods." By generating a fake answer first, you search from within answer-space, dramatically increasing the chance of hitting the right document.',
            bad: '❌ Think about the geometry of embeddings. Question vectors and answer vectors live in different regions of the high-dimensional space. HyDE bridges this gap by generating a hypothetical ANSWER (not question) and searching from that point.'
        }
    };

    const ans = answers[qId];
    if (!ans) return;
    fb.className = `quiz-feedback show ${isCorrect ? 'correct' : 'wrong'}`;
    fb.textContent = isCorrect ? ans.ok : ans.bad;
}
