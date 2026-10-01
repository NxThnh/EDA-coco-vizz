"use client";

import { useEffect, useRef, useState } from "react";

const slideNames = ["Mở đầu", "Câu hỏi", "Dữ liệu", "Domain gap", "Chất lượng", "OCR", "Hallucination", "Vai trò HGAI", "Đánh giá", "Kết luận"];

function Stat({ value, label, tone = "blue" }: { value: string; label: string; tone?: "blue" | "red" | "cream" }) {
  return <div className={`stat ${tone}`}><strong>{value}</strong><span>{label}</span></div>;
}

export default function Home() {
  const deck = useRef<HTMLElement>(null);
  const [current, setCurrent] = useState(0);
  const go = (index: number) => deck.current?.children[Math.max(0, Math.min(slideNames.length - 1, index))]?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(event.key)) { event.preventDefault(); go(current + 1); }
      if (["ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) { event.preventDefault(); go(current - 1); }
      if (event.key === "Home") go(0);
      if (event.key === "End") go(slideNames.length - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current]);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => entry.isIntersecting && setCurrent(Number((entry.target as HTMLElement).dataset.index))), { threshold: 0.65 });
    Array.from(deck.current?.children ?? []).forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  return (
    <main ref={deck} className="deck">
      <section className="slide cover" data-index="0">
        <div className="eyebrow">EDA · COCO × VIZWIZ</div>
        <h1>Dữ liệu nào để<br />VLM nhỏ <em>không đoán?</em></h1>
        <p className="lead">Từ ảnh benchmark đến ảnh hỗ trợ thị giác đời thực</p>
        <div className="cover-strip">
          <img src="/assets/COCO_val2014_000000391895.jpg" alt="Ảnh mẫu COCO: người lái xe máy" />
          <img src="/assets/VizWiz_val_00000001.jpg" alt="Ảnh mẫu VizWiz: chai thuốc" />
          <img src="/assets/VizWiz_val_00000006.jpg" alt="Ảnh VizWiz không thể nhận diện" />
        </div>
        <span className="hint">Dùng phím ← → để trình chiếu</span>
      </section>

      <section className="slide question" data-index="1">
        <div className="kicker">BÀI TOÁN</div>
        <h2>Khi ảnh không đủ bằng chứng,<br /><em>mô hình nên nói gì?</em></h2>
        <div className="scenario">
          <img src="/assets/VizWiz_val_00000001.jpg" alt="Chai thuốc NIGHT TIME trong VizWiz" />
          <div className="scenario-copy"><p className="quote">“a person holding a small black bottle of NIGHT TIME”</p><div className="rule" /><p>Đúng khi nhìn thấy.<br /><b>Nguy hiểm khi chỉ suy đoán.</b></p></div>
        </div>
        <div className="source">VizWiz Val · image 23432 · annotation 117163</div>
      </section>

      <section className="slide" data-index="2">
        <div className="kicker">HAI VAI TRÒ DỮ LIỆU</div><h2>Baseline sạch. Stress test thật.</h2>
        <div className="split-grid">
          <article className="dataset-card"><div className="dataset-head"><span>COCO 2014</span><b>01</b></div><img src="/assets/COCO_val2014_000000522418.jpg" alt="Ảnh COCO người phụ nữ cắt bánh" /><div className="dataset-foot"><strong>123.287</strong><span>ảnh chuẩn hóa</span></div></article>
          <article className="dataset-card"><div className="dataset-head"><span>VizWiz</span><b>02</b></div><img src="/assets/VizWiz_val_00000000.jpg" alt="Ảnh VizWiz màn hình máy tính" /><div className="dataset-foot"><strong>39.181</strong><span>ảnh hỗ trợ đời thực</span></div></article>
        </div>
      </section>

      <section className="slide chart-slide" data-index="3">
        <div className="kicker">DOMAIN GAP · HÌNH HỌC ẢNH</div>
        <div className="chart-layout"><div><h2>Cùng là ảnh.<br />Khác cách nhìn.</h2><div className="stats-row vertical"><Stat value="71,71%" label="COCO ảnh ngang" /><Stat value="92,86%" label="VizWiz ảnh dọc" tone="red" /></div><p className="note">Center crop có thể cắt mất nhãn, chữ hoặc vật thể nằm lệch khung.</p></div><img className="chart" src="/assets/02_resolution_aspect_ratio.png" alt="Biểu đồ EDA độ phân giải và tỷ lệ khung hình" /></div>
      </section>

      <section className="slide dark" data-index="4">
        <div className="kicker">DOMAIN GAP · CHẤT LƯỢNG</div><h2>Nhiều pixel hơn.<br /><em>Không có nghĩa nhìn rõ hơn.</em></h2>
        <div className="quality-grid"><Stat value="1,57 MP" label="VizWiz · độ phân giải TB" tone="cream" /><Stat value="400" label="VizWiz · độ nét Laplacian" tone="cream" /><Stat value="1.082" label="COCO · độ nét Laplacian" tone="cream" /></div>
        <div className="blur-compare"><img src="/assets/COCO_val2014_000000391895.jpg" alt="Ảnh COCO rõ nét" /><img src="/assets/VizWiz_val_00000013.jpg" alt="Ảnh VizWiz bị mờ" /></div>
        <p className="footnote">Laplacian variance là chỉ báo độ sắc nét, không phải thước đo chất lượng tuyệt đối.</p>
      </section>

      <section className="slide" data-index="5">
        <div className="kicker">NHU CẦU ĐỌC CHỮ</div><div className="big-number">73,72%</div><h2 className="compact">ảnh VizWiz có chữ</h2>
        <div className="ocr-band"><img src="/assets/VizWiz_val_00000001.jpg" alt="Nhãn thuốc trong ảnh VizWiz" /><div><span>SCREEN</span><span>BOTTLE</span><span>LABEL</span><span>MEDICINE</span></div></div>
        <p className="note center">Nhận ra “chai” là chưa đủ. Mô hình phải đọc được nhãn — hoặc thừa nhận không đọc được.</p>
      </section>

      <section className="slide failure" data-index="6">
        <div className="kicker">FAILURE MODE</div><div className="failure-layout"><div className="failure-image"><img src="/assets/VizWiz_val_00000006.jpg" alt="Ảnh VizWiz quá kém để nhận diện" /><span>5 / 5 annotators</span></div><div><div className="big-number red-text">32,62%</div><h2>có dấu hiệu<br />không thể nhận diện</h2><p className="quote small">“Quality issues are too severe to recognize visual content.”</p></div></div>
        <div className="source">VizWiz Val: 2.528 / 7.750 ảnh có ít nhất 1/5 caption báo lỗi · 192 ảnh có 5/5</div>
      </section>

      <section className="slide hgai" data-index="7">
        <div className="kicker">HGAI CAN THIỆP Ở ĐÂU?</div><h2>Không chỉ tạo câu trả lời.<br /><em>Kiểm tra bằng chứng.</em></h2>
        <div className="pipeline" aria-label="Quy trình HGAI"><div><b>01</b><span>Đánh giá<br />chất lượng ảnh</span></div><i>→</i><div><b>02</b><span>Grounding<br />ảnh + OCR</span></div><i>→</i><div><b>03</b><span>Trả lời<br />hoặc từ chối</span></div></div>
        <div className="safe-answer">“Ảnh quá mờ để xác định chính xác. Vui lòng chụp lại gần hơn và đủ sáng.”</div>
      </section>

      <section className="slide" data-index="8">
        <div className="kicker">THIẾT KẾ ĐÁNH GIÁ</div><h2>Đo đúng điều ta muốn sửa.</h2>
        <div className="evaluation"><div><span>01</span><b>Baseline</b><p>VLM nhỏ trên COCO → VizWiz</p></div><div><span>02</span><b>Can thiệp</b><p>Thêm HGAI / quality-aware grounding</p></div><div><span>03</span><b>So sánh</b><p>Hallucination · accuracy · refusal</p></div></div>
        <div className="metric-strip"><span>Object hallucination</span><span>OCR hallucination</span><span>Correct refusal</span><span>False refusal</span></div>
      </section>

      <section className="slide conclusion" data-index="9">
        <div className="kicker">KẾT LUẬN</div><h2>COCO dạy mô hình <em>nhìn.</em><br />VizWiz buộc mô hình biết<br /><em>khi nào không nên đoán.</em></h2>
        <div className="takeaway"><span>Baseline chuẩn</span><i>+</i><span>Stress test thật</span><i>+</i><span>HGAI kiểm chứng</span></div>
        <p className="closing">Mục tiêu: VLM nhỏ trả lời đúng khi có bằng chứng — và từ chối an toàn khi không có.</p>
      </section>

      <nav className="nav" aria-label="Điều hướng slide"><button onClick={() => go(current - 1)} aria-label="Slide trước">←</button><button className="counter" onClick={() => go((current + 1) % slideNames.length)} aria-label="Slide tiếp theo"><span>{String(current + 1).padStart(2, "0")}</span> / {String(slideNames.length).padStart(2, "0")}</button><button onClick={() => go(current + 1)} aria-label="Slide tiếp theo">→</button></nav>
    </main>
  );
}
