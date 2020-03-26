<template>
  <div class="post-container">
    <router-link to="/section1/">
      <div class="post-card">
        <img class="article-image" src="../public/images/cover.jpg" />
        <div class="description">
          Section1
        </div>
      </div>
    </router-link>
    <router-link to="/section2/">
      <div class="post-card">
        <img class="article-image" src="../public/images/cover.jpg" />
        <div class="description">
          Section2
        </div>
      </div>
    </router-link>
    <router-link to="/section3/">
      <div class="post-card">
        <img class="article-image" src="../public/images/cover.jpg" />
        <div class="description">
          Section3
        </div>
      </div>
    </router-link>
    <router-link to="/section6/">
      <div class="post-card">
        <img class="article-image" src="../public/images/cover.jpg" />
        <div class="description">
          Section6
        </div>
      </div>
    </router-link>
  </div>
</template>

<script>
export default {};
</script>
<style scoped>
.post-container {
  display: flex;
  flex-wrap: wrap;
  width: 100%;
}
.post-card {
  width: 600px;
  height: 150px;
  margin: 10px;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 10px;
  display: flex;
  align-items: center;
}
.article-image {
  height: 100%;
}
.description {
  width: 100%;
  display: flex;
  justify-content: center;
}
</style>
