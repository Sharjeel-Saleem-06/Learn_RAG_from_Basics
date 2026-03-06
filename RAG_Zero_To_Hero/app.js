/* ============================================
   RAG ZERO TO HERO — Interactivity
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
    // Load advanced content
    loadAdvancedContent();

    // Scroll animations
    setupScrollAnimations();

    // Active nav link on scroll
    setupActiveNav();
});

// --- Load Advanced Content ---
async function loadAdvancedContent() {
    const container = document.getElementById('content-part2');
    try {
        const response = await fetch('advanced.html');
        if (response.ok) {
            const html = await response.text();
            container.innerHTML = html;
            // Re-setup after content loaded
            setupScrollAnimations();
        }
    } catch (e) {
        // Fallback: advanced.html might not load via file:// protocol
        // In that case, user should use a local server
        container.innerHTML = `
      <div class="container" style="padding:60px 20px;text-align:center;">
        <div class="concept-card">
          <h3>📄 Load Advanced Sections</h3>
          <p>To view advanced sections (HyDE, Re-ranking, Agentic RAG, Evaluation), 
          please open this file using a local server:</p>
          <div class="code-block">
            <code><span class="comment"># Option 1: Python HTTP Server</span>
python3 -m http.server 8081

<span class="comment"># Then open: http://localhost:8081</span>

<span class="comment"># Option 2: VS Code Live Server extension</span>
<span class="comment"># Right-click index.html → Open with Live Server</span></code>
          </div>
          <p style="margin-top:16px;color:var(--text-secondary);">
            This is needed because browsers block loading local files (CORS policy).<br>
            The content is in <code>advanced.html</code> in the same folder.
          </p>
        </div>
      </div>`;
    }
}

// --- Scroll Animations (Intersection Observer) ---
function setupScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.concept-card, .mini-card, .section-header').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// --- Active Nav Link ---
function setupActiveNav() {
    const sections = document.querySelectorAll('.section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const top = section.offsetTop - 120;
            if (window.scrollY >= top) {
                current = section.getAttribute('id');
            }
        });
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });
}
